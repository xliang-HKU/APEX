"""Finite-temperature elastic constants from LAMMPS NPT/NVT averaging."""

import logging
import os
import re
from typing import Dict, List, Tuple

from monty.serialization import dumpfn, loadfn
from pymatgen.analysis.elasticity.elastic import ElasticTensor
from pymatgen.analysis.elasticity.strain import Strain
from pymatgen.analysis.elasticity.stress import Stress
from pymatgen.core.structure import Structure

from apex.core.calculator.lib import vasp_utils
from apex.core.property.Property import Property
from apex.core.refine import make_refine
from dflow.python import upload_packages

upload_packages.append(__file__)

DEFAULT_SUPERCELL = [2, 2, 2]
DEFAULT_CAL_SETTING: Dict[str, int | float | List[int]] = {
    "temperature": [200, 400, 600, 800],
    "strain": 0.01,
    "strain_components": [0, 1, 2, 3, 4, 5],
    "equi_step": 16000,
    "deform_equi_step": 16000,
    "N_every": 100,
    "N_repeat": 40,
    "N_freq": 4000,
    "ave_step": 16000,
    "timestep": 0.001,
    "tdamp": 0.1,
    "pdamp": 1.0,
    "seed": 12345,
}

COMPONENT_LABELS = {
    0: "xx",
    1: "yy",
    2: "zz",
    3: "yz",
    4: "xz",
    5: "xy",
}

COMPONENT_ALIASES = {
    "xx": 0,
    "11": 0,
    "e11": 0,
    "yy": 1,
    "22": 1,
    "e22": 1,
    "zz": 2,
    "33": 2,
    "e33": 2,
    "yz": 3,
    "zy": 3,
    "23": 3,
    "32": 3,
    "e23": 3,
    "e32": 3,
    "xz": 4,
    "zx": 4,
    "13": 4,
    "31": 4,
    "e13": 4,
    "e31": 4,
    "xy": 5,
    "yx": 5,
    "12": 5,
    "21": 5,
    "e12": 5,
    "e21": 5,
}


class FiniteTela(Property):
    """
    Generate LAMMPS tasks for finite-temperature elastic constants.

    Each temperature includes one reference task to sample the equilibrium
    stress after the NPT stage and +/- strained tasks for the six Voigt
    components, followed by NVT averaging of the stresses.
    """

    def __init__(self, parameter: Dict, inter_param: Dict | None = None):
        if parameter.get("reproduce", False):
            raise NotImplementedError("FiniteTela does not support reproduce mode.")

        if inter_param is not None and inter_param.get("type") in ["vasp", "abacus"]:
            raise TypeError("FiniteTela supports only LAMMPS calculations.")

        parameter.setdefault("cal_setting", {})
        for key, val in DEFAULT_CAL_SETTING.items():
            parameter["cal_setting"].setdefault(key, val)

        parameter.setdefault("supercell_size", DEFAULT_SUPERCELL)
        self.supercell_size = parameter["supercell_size"]

        parameter.setdefault("modulus_type", "voigt")
        self.modulus_type = parameter["modulus_type"]

        self.cal_setting = parameter["cal_setting"]
        self.strain_magnitude = float(self.cal_setting["strain"])
        self.strain_components = self._normalize_components(
            self.cal_setting["strain_components"]
        )

        if "init_from_suffix" in parameter and "output_suffix" in parameter:
            self.init_from_suffix = parameter["init_from_suffix"]
            self.output_suffix = parameter["output_suffix"]

        parameter["cal_type"] = "npt+deform+nvt+ave/time"
        self.parameter = parameter
        self.inter_param = inter_param or {"type": "lammps"}

    def make_confs(self, path_to_work: str, path_to_equi: str, refine: bool = False):
        path_to_work = os.path.abspath(path_to_work)
        os.makedirs(path_to_work, exist_ok=True)
        path_to_equi = os.path.abspath(path_to_equi)

        cwd = os.getcwd()
        if refine:
            task_list = self._make_refine(path_to_work)
        else:
            task_list = self._make_fresh_tasks(path_to_work, path_to_equi)
        os.chdir(cwd)
        return task_list

    def post_process(self, task_list):
        pass

    def task_type(self):
        return self.parameter["type"]

    def task_param(self):
        return self.parameter

    def _compute_lower(self, output_file, all_tasks, all_res):
        output_file = os.path.abspath(output_file)
        res_data: Dict[str, Dict] = {}
        ptr_data = os.path.dirname(output_file) + "\n"

        grouped_tasks: Dict[str, List[Tuple[str, Dict]]] = {}
        for task_dir in all_tasks:
            meta = loadfn(os.path.join(task_dir, "FiniteTela.json"))
            temp_key = str(meta["temperature"])
            grouped_tasks.setdefault(temp_key, []).append((task_dir, meta))

        for temp_key in sorted(grouped_tasks, key=float):
            ref_stress = None
            lst_strain = []
            lst_stress = []

            for task_dir, meta in grouped_tasks[temp_key]:
                avg_stress = self._average_stress(task_dir)
                stress = Stress(avg_stress)
                stress *= -1000

                if meta["is_reference"]:
                    ref_stress = stress
                    continue

                strain = loadfn(os.path.join(task_dir, "strain.json"))
                lst_strain.append(strain)
                lst_stress.append(stress)

            if ref_stress is None:
                raise RuntimeError(f"No reference task found for temperature {temp_key}")

            et = ElasticTensor.from_independent_strains(
                lst_strain, lst_stress, eq_stress=ref_stress, vasp=False
            )
            temp_res = self._tensor_to_result(et, float(temp_key), ref_stress)
            res_data[temp_key] = temp_res
            ptr_data += self._format_temperature_block(temp_res)

        dumpfn(res_data, output_file, indent=4)
        return res_data, ptr_data

    def _make_refine(self, path_to_work: str) -> List[str]:
        logging.info("FiniteTela refine starts")
        task_list = make_refine(
            self.init_from_suffix,
            self.output_suffix,
            path_to_work,
        )
        init_from_path = re.sub(
            self.output_suffix[::-1],
            self.init_from_suffix[::-1],
            path_to_work[::-1],
            count=1,
        )[::-1]
        for task_name in map(os.path.basename, task_list):
            init_task = os.path.join(init_from_path, task_name)
            out_task = os.path.join(path_to_work, task_name)
            for file_name in [
                "FiniteTela.json",
                "strain.json",
                "variable_FiniteTela.in",
                "deform_FiniteTela.in",
            ]:
                self._symlink_task_file(init_task, out_task, file_name)
        return task_list

    def _make_fresh_tasks(self, path_to_work: str, path_to_equi: str) -> List[str]:
        if self.inter_param["type"] in ["vasp", "abacus"]:
            raise TypeError("FiniteTela only supports LAMMPS calculation")

        equi_contcar = os.path.join(path_to_equi, "CONTCAR")
        if not os.path.exists(equi_contcar):
            raise RuntimeError("please do relaxation first")

        ptypes = vasp_utils.get_poscar_types(equi_contcar)
        structure = Structure.from_file(equi_contcar)

        task_list: List[str] = []
        task_idx = 0
        for temp in self.cal_setting["temperature"]:
            task_dir = os.path.join(path_to_work, f"task.{task_idx:06d}")
            os.makedirs(task_dir, exist_ok=True)
            self._write_task(task_dir, structure, ptypes, temp, None, 0.0)
            task_list.append(task_dir)
            task_idx += 1

            for component in self.strain_components:
                for sign in [-1.0, 1.0]:
                    strain_value = sign * self.strain_magnitude
                    task_dir = os.path.join(path_to_work, f"task.{task_idx:06d}")
                    os.makedirs(task_dir, exist_ok=True)
                    self._write_task(
                        task_dir, structure, ptypes, temp, component, strain_value
                    )
                    task_list.append(task_dir)
                    task_idx += 1

        return task_list

    def _normalize_components(self, components) -> List[int]:
        norm_components = []
        for component in components:
            if isinstance(component, int):
                if component not in COMPONENT_LABELS:
                    raise ValueError(f"Unsupported strain component index: {component}")
                norm_components.append(component)
                continue

            key = str(component).strip().lower()
            if key not in COMPONENT_ALIASES:
                raise ValueError(f"Unsupported strain component label: {component}")
            norm_components.append(COMPONENT_ALIASES[key])

        if not norm_components:
            raise ValueError("strain_components cannot be empty")
        return list(dict.fromkeys(norm_components))

    def _symlink_task_file(self, init_task: str, out_task: str, file_name: str):
        os.makedirs(out_task, exist_ok=True)
        dst = os.path.join(out_task, file_name)
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        src = os.path.join(init_task, file_name)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing refine input file: {src}")
        os.symlink(os.path.relpath(src, out_task), dst)

    def _write_task(
        self,
        task_dir: str,
        structure: Structure,
        ptypes,
        temp: float,
        strain_component: int | None,
        strain_value: float,
    ):
        os.chdir(task_dir)
        for fname in [
            "INCAR",
            "POTCAR",
            "POSCAR",
            "conf.lmp",
            "in.lammps",
            "STRU",
            "strain.json",
            "FiniteTela.json",
            "variable_FiniteTela.in",
            "deform_FiniteTela.in",
        ]:
            if os.path.exists(fname):
                os.remove(fname)

        structure.to("POSCAR.tmp", "POSCAR")
        vasp_utils.regulate_poscar("POSCAR.tmp", "POSCAR")
        vasp_utils.sort_poscar("POSCAR", "POSCAR", ptypes)
        os.remove("POSCAR.tmp")

        strain_voigt = [0.0] * 6
        if strain_component is not None:
            strain_voigt[strain_component] = strain_value
            strain_label = COMPONENT_LABELS[strain_component]
        else:
            strain_label = "reference"

        task_meta = {
            "temperature": float(temp),
            "supercell_size": self.supercell_size,
            "strain_component": strain_component,
            "strain_label": strain_label,
            "strain_value": float(strain_value),
            "is_reference": strain_component is None,
        }
        dumpfn(task_meta, "FiniteTela.json", indent=4)
        dumpfn(Strain.from_voigt(strain_voigt), "strain.json", indent=4)

        with open("variable_FiniteTela.in", "w") as fp:
            fp.write(self._variable(temp))
        with open("deform_FiniteTela.in", "w") as fp:
            fp.write(self._deform(strain_component, strain_value))

    def _average_stress(self, task_dir: str):
        stress_sum = [[0.0, 0.0, 0.0] for _ in range(3)]
        count = 0
        stress_file = os.path.join(task_dir, "average_stress.txt")
        with open(stress_file, "r") as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 7:
                    continue
                _, pxx, pyy, pzz, pxy, pxz, pyz = map(float, parts)
                stress_sum[0][0] += pxx
                stress_sum[1][1] += pyy
                stress_sum[2][2] += pzz
                stress_sum[0][1] += pxy
                stress_sum[1][0] += pxy
                stress_sum[0][2] += pxz
                stress_sum[2][0] += pxz
                stress_sum[1][2] += pyz
                stress_sum[2][1] += pyz
                count += 1

        if count == 0:
            raise RuntimeError(f"No averaged stress data found in {stress_file}")

        for ii in range(3):
            for jj in range(3):
                stress_sum[ii][jj] /= count * 1000.0
        return stress_sum

    def _tensor_to_result(
        self, elastic_tensor: ElasticTensor, temperature: float, ref_stress: Stress
    ) -> Dict:
        res_data: Dict[str, float | List[List[float]]] = {
            "temperature": temperature,
            "elastic_tensor": [],
            "equilibrium_stress": [],
        }

        tensor_gpa = []
        for ii in range(6):
            row = []
            for jj in range(6):
                row.append(float(elastic_tensor.voigt[ii][jj] / 1e4))
            tensor_gpa.append(row)
        res_data["elastic_tensor"] = tensor_gpa

        for ii in range(3):
            row = []
            for jj in range(3):
                row.append(float(ref_stress[ii][jj] / 1e4))
            res_data["equilibrium_stress"].append(row)

        if self.modulus_type == "voigt":
            bulk_modulus = elastic_tensor.k_voigt / 1e4
            shear_modulus = elastic_tensor.g_voigt / 1e4
        elif self.modulus_type == "reuss":
            bulk_modulus = elastic_tensor.k_reuss / 1e4
            shear_modulus = elastic_tensor.g_reuss / 1e4
        elif self.modulus_type == "vrh":
            bulk_modulus = elastic_tensor.k_vrh / 1e4
            shear_modulus = elastic_tensor.g_vrh / 1e4
        else:
            raise ValueError(f"Unsupported modulus_type: {self.modulus_type}")

        youngs_modulus = 9 * bulk_modulus * shear_modulus / (
            3 * bulk_modulus + shear_modulus
        )
        poisson_ratio = 0.5 * (3 * bulk_modulus - 2 * shear_modulus) / (
            3 * bulk_modulus + shear_modulus
        )

        res_data["B"] = float(bulk_modulus)
        res_data["G"] = float(shear_modulus)
        res_data["E"] = float(youngs_modulus)
        res_data["u"] = float(poisson_ratio)
        return res_data

    def _format_temperature_block(self, res_data: Dict) -> str:
        ptr_data = f"Temperature: {res_data['temperature']:.2f} K\n"
        ptr_data += "# Equilibrium stress tensor (GPa)\n"
        for row in res_data["equilibrium_stress"]:
            ptr_data += " ".join(f"{value:9.4f}" for value in row) + "\n"
        ptr_data += "# Elastic tensor (GPa)\n"
        for row in res_data["elastic_tensor"]:
            ptr_data += " ".join(f"{value:9.2f}" for value in row) + "\n"
        ptr_data += f"# Bulk   Modulus B = {res_data['B']:.2f} GPa\n"
        ptr_data += f"# Shear  Modulus G = {res_data['G']:.2f} GPa\n"
        ptr_data += f"# Youngs Modulus E = {res_data['E']:.2f} GPa\n"
        ptr_data += f"# Poisson Ratio u = {res_data['u']:.4f}\n\n"
        return ptr_data

    def _variable(self, temp: float) -> str:
        return (
            " # variable_FiniteTela.in \n"
            f"variable temperature equal {temp:.2f}\n"
            f"variable nx equal {self.supercell_size[0]}\n"
            f"variable ny equal {self.supercell_size[1]}\n"
            f"variable nz equal {self.supercell_size[2]}\n"
            f"variable equi_step equal {self.cal_setting['equi_step']}\n"
            f"variable deform_equi_step equal {self.cal_setting['deform_equi_step']}\n"
            f"variable N_every equal {self.cal_setting['N_every']}\n"
            f"variable N_repeat equal {self.cal_setting['N_repeat']}\n"
            f"variable N_freq equal {self.cal_setting['N_freq']}\n"
            f"variable ave_step equal {self.cal_setting['ave_step']}\n"
            f"variable timestep equal {self.cal_setting['timestep']}\n"
            f"variable tdamp equal {self.cal_setting['tdamp']}\n"
            f"variable pdamp equal {self.cal_setting['pdamp']}\n"
            f"variable seed equal {self.cal_setting['seed']}\n"
        )

    def _deform(self, strain_component: int | None, strain_value: float) -> str:
        header = (
            " # deform_FiniteTela.in \n"
            f"variable strain equal {strain_value:.8f}\n"
            "change_box all triclinic\n"
        )
        if strain_component is None:
            return header + "# reference task: no applied strain\n"

        if strain_component == 0:
            return (
                header
                + f"change_box all x scale {1.0 + strain_value:.8f} remap units box\n"
            )
        if strain_component == 1:
            return (
                header
                + f"change_box all y scale {1.0 + strain_value:.8f} remap units box\n"
            )
        if strain_component == 2:
            return (
                header
                + f"change_box all z scale {1.0 + strain_value:.8f} remap units box\n"
            )
        if strain_component == 3:
            return (
                header
                + "variable tilt_delta equal ${strain}*lz\n"
                + "change_box all yz delta ${tilt_delta} remap units box\n"
            )
        if strain_component == 4:
            return (
                header
                + "variable tilt_delta equal ${strain}*lz\n"
                + "change_box all xz delta ${tilt_delta} remap units box\n"
            )
        if strain_component == 5:
            return (
                header
                + "variable tilt_delta equal ${strain}*ly\n"
                + "change_box all xy delta ${tilt_delta} remap units box\n"
            )

        raise ValueError(f"Unsupported strain component: {strain_component}")
