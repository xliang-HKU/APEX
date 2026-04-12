import glob
import os
import shutil
import sys
import tempfile
import unittest

from monty.serialization import dumpfn, loadfn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
__package__ = "tests"

from apex.core.calculator.Lammps import Lammps
from apex.core.property.FiniteBulk import FiniteBulk


class TestFiniteBulk(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.dirname(__file__)
        base = {
            "structures": ["confs/hcp-Ti"],
            "interaction": {
                "type": "meam_spline",
                "model": os.path.join(self.test_dir, "lammps_input", "Ti.meam.spline"),
                "type_map": {"Ti": 0},
            },
            "properties": [
                {
                    "type": "finitebulk",
                    "supercell_size": [2, 2, 2],
                    "cal_setting": {
                        "temperature": [400],
                        "strain": 0.01,
                        "equi_step": 4000,
                        "deform_equi_step": 4000,
                        "N_every": 100,
                        "N_repeat": 5,
                        "N_freq": 1000,
                        "ave_step": 4000,
                        "seed": 12345,
                    },
                }
            ],
        }

        self.work_root = tempfile.mkdtemp(prefix="apex_finitebulk_", dir="/tmp")
        self.equi_path = os.path.join(
            self.work_root, "confs", "hcp-Ti", "relaxation", "relax_task"
        )
        self.source_path = os.path.join(self.test_dir, "equi", "lammps")
        self.target_path = os.path.join(
            self.work_root, "confs", "hcp-Ti", "FiniteBulk_00"
        )

        os.makedirs(self.equi_path, exist_ok=True)

        self.inter_param = base["interaction"]
        self.prop_param = base["properties"]
        self.bulk = FiniteBulk(self.prop_param[0])
        self.lammps = Lammps(
            self.inter_param, os.path.join(self.source_path, "hcp-Ti-CONTCAR")
        )

    def tearDown(self):
        if os.path.exists(self.work_root):
            shutil.rmtree(self.work_root)

    def test_task_type(self):
        self.assertEqual("finitebulk", self.bulk.task_type())

    def test_task_param(self):
        self.assertEqual(self.prop_param[0], self.bulk.task_param())

    def test_make_potential_files(self):
        cwd = os.getcwd()
        abs_equi_path = os.path.abspath(self.equi_path)
        self.lammps.make_potential_files(abs_equi_path)
        self.assertTrue(os.path.islink(os.path.join(self.equi_path, "Ti.meam.spline")))
        self.assertTrue(os.path.isfile(os.path.join(self.equi_path, "inter.json")))
        ret = loadfn(os.path.join(self.equi_path, "inter.json"))
        self.assertEqual(self.inter_param, ret)
        os.chdir(cwd)

    def test_make_confs(self):
        if not os.path.exists(os.path.join(self.equi_path, "CONTCAR")):
            with self.assertRaises(RuntimeError):
                self.bulk.make_confs(self.target_path, self.equi_path)

        shutil.copy(
            os.path.join(self.source_path, "hcp-Ti-CONTCAR"),
            os.path.join(self.equi_path, "CONTCAR"),
        )

        task_list = self.bulk.make_confs(self.target_path, self.equi_path)
        self.assertEqual(len(task_list), 3)

        dfm_dirs = glob.glob(os.path.join(self.target_path, "task.*"))
        dfm_dirs.sort()
        self.assertEqual(len(dfm_dirs), 3)

        ref_meta = loadfn(os.path.join(dfm_dirs[0], "FiniteBulk.json"))
        self.assertTrue(ref_meta["is_reference"])
        self.assertEqual("reference", ref_meta["strain_label"])
        self.assertEqual(0.0, ref_meta["strain_value"])

        with open(os.path.join(dfm_dirs[0], "deform_FiniteBulk.in"), "r") as fp:
            ref_deform = fp.read()
        self.assertIn("reference task: no applied isotropic strain", ref_deform)

        perturbations = set()
        for task_dir in dfm_dirs[1:]:
            self.assertTrue(os.path.isfile(os.path.join(task_dir, "POSCAR")))
            self.assertTrue(os.path.isfile(os.path.join(task_dir, "FiniteBulk.json")))
            self.assertTrue(os.path.isfile(os.path.join(task_dir, "strain.json")))
            self.assertTrue(os.path.isfile(os.path.join(task_dir, "variable_FiniteBulk.in")))
            self.assertTrue(os.path.isfile(os.path.join(task_dir, "deform_FiniteBulk.in")))

            meta = loadfn(os.path.join(task_dir, "FiniteBulk.json"))
            perturbations.add(meta["strain_value"])

        self.assertEqual({-0.01, 0.01}, perturbations)

        with open(os.path.join(dfm_dirs[-1], "deform_FiniteBulk.in"), "r") as fp:
            bulk_deform = fp.read()
        self.assertIn("x scale 1.01000000", bulk_deform)
        self.assertIn("y scale 1.01000000", bulk_deform)
        self.assertIn("z scale 1.01000000", bulk_deform)

    def test_average_pressure_tensor(self):
        task_dir = os.path.join(self.target_path, "task.avg")
        os.makedirs(task_dir, exist_ok=True)

        with open(os.path.join(task_dir, "average_stress.txt"), "w") as fp:
            fp.write("# step pxx pyy pzz pxy pxz pyz\n")
            fp.write("0 1000 2000 3000 400 500 600\n")
            fp.write("1 3000 4000 5000 800 900 1000\n")

        tensor = self.bulk._average_pressure_tensor_bar(task_dir)
        self.assertEqual(
            [
                [2000.0, 600.0, 700.0],
                [600.0, 3000.0, 800.0],
                [700.0, 800.0, 4000.0],
            ],
            tensor,
        )

    def test_compute_lower(self):
        task_dirs = []
        pressure_map = {
            0.0: 0.0,
            -0.01: 1500.0,
            0.01: -1500.0,
        }
        for idx, strain_value in enumerate([0.0, -0.01, 0.01]):
            task_dir = os.path.join(self.target_path, f"task.{idx:06d}")
            os.makedirs(task_dir, exist_ok=True)
            with open(os.path.join(task_dir, "average_stress.txt"), "w") as fp:
                fp.write("# step pxx pyy pzz pxy pxz pyz\n")
                p = pressure_map[strain_value]
                fp.write(f"0 {p} {p} {p} 0 0 0\n")

            dumpfn(
                {
                    "temperature": 400.0,
                    "supercell_size": [2, 2, 2],
                    "strain_label": "reference" if strain_value == 0.0 else "bulk",
                    "strain_value": strain_value,
                    "is_reference": strain_value == 0.0,
                },
                os.path.join(task_dir, "FiniteBulk.json"),
                indent=4,
            )
            task_dirs.append(task_dir)

        output_file = os.path.join(self.target_path, "result.json")
        res_data, ptr_data = self.bulk._compute_lower(output_file, task_dirs, [])

        temp_key = "400.0"
        self.assertIn(temp_key, res_data)
        self.assertAlmostEqual(5.0, res_data[temp_key]["B"], places=3)
        self.assertAlmostEqual(0.0, res_data[temp_key]["equilibrium_pressure"], places=6)
        self.assertIn("Bulk Modulus B", ptr_data)

    def test_forward_common_files(self):
        fc_files = [
            "in.lammps",
            "variable_FiniteBulk.in",
            "deform_FiniteBulk.in",
            "Ti.meam.spline",
        ]
        self.assertEqual(
            self.lammps.forward_common_files(self.prop_param[0]["type"]), fc_files
        )

    def test_backward_files(self):
        backward_files = ["log.lammps", "outlog", "dump.relax", "average_stress.txt"]
        self.assertEqual(
            self.lammps.backward_files(self.prop_param[0]["type"]), backward_files
        )
