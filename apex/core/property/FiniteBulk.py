"""Finite-temperature bulk modulus from isotropic LAMMPS deformations."""

import os
import re
from typing import Dict, List, Tuple

from monty.serialization import dumpfn, loadfn
from pymatgen.analysis.elasticity.strain import Strain
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


class FiniteBulk(Property):
    """
    Measure the finite-temperature bulk modulus with isotropic +/- deformations.

    Each temperature produces one equilibrium reference task and two deformed
    tasks with equal strain applied along x, y, and z. The post-process stage
    reconstructs the hydrostatic pressure response and converts the centered
    pressure-volume slope into B(T).
    """

    def __init__(self, parameter: Dict, inter_param: Dict | None = None):
        if parameter.get("reproduce", False):
            raise NotImplementedError("FiniteBulk does not support reproduce mode.")

        if inter_param is not None and inter_param.get("type") in ["vasp", "abacus"]:
            raise TypeError("FiniteBulk supports only LAMMPS calculations.")

        parameter.setdefault("cal_setting", {})
        for key, val in DEFAULT_CAL_SETTING.items():
            parameter["cal_setting"].setdefault(key, val)

        parameter.setdefault("supercell_size", DEFAULT_SUPERCELL)
        self.supercell_size = parameter["supercell_size"]
        self.cal_setting = parameter["cal_setting"]
        self.strain_magnitude = float(self.cal_setting["strain"])

        if "init_from_suffix" in parameter and "output_suffix" in parameter:
            self.init_from_suffix = parameter["init_from_suffix"]
            self.output_suffix = parameter["output_suffix"]

        parameter["cal_type"] = "npt+bulkdeform+nvt+ave/time"
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
            meta = loadfn(os.path.join(task_dir, "FiniteBulk.json"))
            temp_key = str(meta["temperature"])
            grouped_tasks.setdefault(temp_key, []).append((task_dir, meta))

        for temp_key in sorted(grouped_tasks, key=float):
            ref_pressure = None
            ref_stress = None
            plus_point = None
            minus_point = None

            for task_dir, meta in grouped_tasks[temp_key]:
                avg_pressure_tensor = self._average_pressure_tensor_bar(task_dir)
                pressure_gpa = self._pressure_gpa(avg_pressure_tensor)
                stress_gpa = self._stress_gpa(avg_pressure_tensor)

                if meta["is_reference"]:
                    ref_pressure = pressure_gpa
                    ref_stress = stress_gpa
                    continue

                volumetric_strain = self._volumetric_strain(meta["strain_value"])
                point = (volumetric_strain, pressure_gpa)
                if meta["strain_value"] > 0:
                    plus_point = point
                else:
                    minus_point = point

            if ref_pressure is None or ref_stress is None:
                raise RuntimeError(f"No reference task found for temperature {temp_key}")
            if plus_point is None or minus_point is None:
                raise RuntimeError(
                    f"Missing +/- isotropic deformation tasks for temperature {temp_key}"
                )

            bulk_modulus = -(
                (plus_point[1] - minus_point[1]) / (plus_point[0] - minus_point[0])
            )

            temp_res = {
                "temperature": float(temp_key),
                "B": float(bulk_modulus),
                "strain": self.strain_magnitude,
                "volumetric_strain_plus": float(plus_point[0]),
                "volumetric_strain_minus": float(minus_point[0]),
                "pressure_plus": float(plus_point[1]),
                "pressure_minus": float(minus_point[1]),
                "equilibrium_pressure": float(ref_pressure),
                "equilibrium_stress": ref_stress,
            }
            res_data[temp_key] = temp_res
            ptr_data += self._format_temperature_block(temp_res)

        dumpfn(res_data, output_file, indent=4)
        return res_data, ptr_data

    def _make_refine(self, path_to_work: str) -> List[str]:
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
                "FiniteBulk.json",
                "strain.json",
                "variable_FiniteBulk.in",
                "deform_FiniteBulk.in",
            ]:
                self._symlink_task_file(init_task, out_task, file_name)
        return task_list

    def _make_fresh_tasks(self, path_to_work: str, path_to_equi: str) -> List[str]:
        if self.inter_param["type"] in ["vasp", "abacus"]:
            raise TypeError("FiniteBulk only supports LAMMPS calculation")

        equi_contcar = os.path.join(path_to_equi, "CONTCAR")
        if not os.path.exists(equi_contcar):
            raise RuntimeError("please do relaxation first")

        ptypes = vasp_utils.get_poscar_types(equi_contcar)
        structure = Structure.from_file(equi_contcar)

        task_list: List[str] = []
        task_idx = 0
        for temp in self.cal_setting["temperature"]:
            for strain_value in [0.0, -self.strain_magnitude, self.strain_magnitude]:
                task_dir = os.path.join(path_to_work, f"task.{task_idx:06d}")
                os.makedirs(task_dir, exist_ok=True)
                self._write_task(task_dir, structure, ptypes, temp, strain_value)
                task_list.append(task_dir)
                task_idx += 1

        return task_list

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
            "FiniteBulk.json",
            "variable_FiniteBulk.in",
            "deform_FiniteBulk.in",
        ]:
            if os.path.exists(fname):
                os.remove(fname)

        structure.to("POSCAR.tmp", "POSCAR")
        vasp_utils.regulate_poscar("POSCAR.tmp", "POSCAR")
        vasp_utils.sort_poscar("POSCAR", "POSCAR", ptypes)
        os.remove("POSCAR.tmp")

        task_meta = {
            "temperature": float(temp),
            "supercell_size": self.supercell_size,
            "strain_label": "reference" if strain_value == 0.0 else "bulk",
            "strain_value": float(strain_value),
            "is_reference": strain_value == 0.0,
        }
        dumpfn(task_meta, "FiniteBulk.json", indent=4)
        dumpfn(Strain.from_voigt([strain_value, strain_value, strain_value, 0, 0, 0]), "strain.json", indent=4)

        with open("variable_FiniteBulk.in", "w") as fp:
            fp.write(self._variable(temp))
        with open("deform_FiniteBulk.in", "w") as fp:
            fp.write(self._deform(strain_value))

    def _average_pressure_tensor_bar(self, task_dir: str):
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
                stress_sum[ii][jj] /= count
        return stress_sum

    @staticmethod
    def _stress_gpa(pressure_tensor_bar):
        return [
            [float(-pressure_tensor_bar[ii][jj] / 10000.0) for jj in range(3)]
            for ii in range(3)
        ]

    @staticmethod
    def _pressure_gpa(pressure_tensor_bar):
        hydrostatic_bar = (
            pressure_tensor_bar[0][0]
            + pressure_tensor_bar[1][1]
            + pressure_tensor_bar[2][2]
        ) / 3.0
        return float(hydrostatic_bar / 10000.0)

    @staticmethod
    def _volumetric_strain(strain_value: float):
        scale = 1.0 + float(strain_value)
        return float(scale**3 - 1.0)

    def _format_temperature_block(self, res_data: Dict) -> str:
        ptr_data = f"Temperature: {res_data['temperature']:.2f} K\n"
        ptr_data += "# Equilibrium stress tensor (GPa)\n"
        for row in res_data["equilibrium_stress"]:
            ptr_data += " ".join(f"{value:9.4f}" for value in row) + "\n"
        ptr_data += f"# Equilibrium pressure = {res_data['equilibrium_pressure']:.4f} GPa\n"
        ptr_data += f"# Pressure (+strain)  = {res_data['pressure_plus']:.4f} GPa\n"
        ptr_data += f"# Pressure (-strain)  = {res_data['pressure_minus']:.4f} GPa\n"
        ptr_data += (
            f"# Volumetric strain (+/-) = "
            f"{res_data['volumetric_strain_plus']:.6f} / "
            f"{res_data['volumetric_strain_minus']:.6f}\n"
        )
        ptr_data += f"# Bulk Modulus B = {res_data['B']:.4f} GPa\n\n"
        return ptr_data

    def _variable(self, temp: float) -> str:
        return (
            " # variable_FiniteBulk.in \n"
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

    @staticmethod
    def _deform(strain_value: float) -> str:
        header = (
            " # deform_FiniteBulk.in \n"
            f"variable strain equal {strain_value:.8f}\n"
        )
        if strain_value == 0.0:
            return header + "# reference task: no applied isotropic strain\n"

        scale = 1.0 + strain_value
        return (
            header
            + "change_box all triclinic\n"
            + (
                "change_box all "
                f"x scale {scale:.8f} "
                f"y scale {scale:.8f} "
                f"z scale {scale:.8f} remap units box\n"
            )
        )
