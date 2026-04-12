import glob
import os
import shutil
import sys
import unittest

from monty.serialization import loadfn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
__package__ = "tests"

from apex.core.calculator.Lammps import Lammps
from apex.core.property.FiniteTela import FiniteTela


class TestFiniteTela(unittest.TestCase):
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
                    "type": "finitetela",
                    "supercell_size": [2, 2, 2],
                    "cal_setting": {
                        "temperature": [400],
                        "strain": 0.01,
                        "strain_components": ["xx", "yz"],
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

        self.equi_path = os.path.join(
            self.test_dir, "confs", "hcp-Ti", "relaxation", "relax_task"
        )
        self.source_path = os.path.join(self.test_dir, "equi", "lammps")
        self.target_path = os.path.join(self.test_dir, "confs", "hcp-Ti", "FiniteTela_00")

        if not os.path.exists(self.equi_path):
            os.makedirs(self.equi_path)

        self.confs = base["structures"]
        self.inter_param = base["interaction"]
        self.prop_param = base["properties"]

        self.finite = FiniteTela(self.prop_param[0])
        self.lammps = Lammps(
            self.inter_param, os.path.join(self.source_path, "hcp-Ti-CONTCAR")
        )

    def tearDown(self):
        if os.path.exists(os.path.abspath(os.path.join(self.equi_path, ".."))):
            shutil.rmtree(os.path.abspath(os.path.join(self.equi_path, "..")))
        if os.path.exists(self.equi_path):
            shutil.rmtree(self.equi_path)
        if os.path.exists(self.target_path):
            shutil.rmtree(self.target_path)

    def test_task_type(self):
        self.assertEqual("finitetela", self.finite.task_type())

    def test_task_param(self):
        self.assertEqual(self.prop_param[0], self.finite.task_param())

    def test_normalize_components_aliases(self):
        components = self.finite._normalize_components(["xx", "e32", "21", 4])
        self.assertEqual([0, 3, 5, 4], components)

    def test_normalize_components_invalid(self):
        with self.assertRaises(ValueError):
            self.finite._normalize_components(["bad-component"])

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
                self.finite.make_confs(self.target_path, self.equi_path)

        shutil.copy(
            os.path.join(self.source_path, "hcp-Ti-CONTCAR"),
            os.path.join(self.equi_path, "CONTCAR"),
        )

        task_list = self.finite.make_confs(self.target_path, self.equi_path)
        self.assertEqual(len(task_list), 5)

        dfm_dirs = glob.glob(os.path.join(self.target_path, "task.*"))
        dfm_dirs.sort()
        self.assertEqual(len(dfm_dirs), 5)

        ref_meta = loadfn(os.path.join(dfm_dirs[0], "FiniteTela.json"))
        self.assertTrue(ref_meta["is_reference"])
        self.assertEqual("reference", ref_meta["strain_label"])
        self.assertEqual(0.0, ref_meta["strain_value"])
        self.assertIsNone(ref_meta["strain_component"])

        with open(os.path.join(dfm_dirs[0], "deform_FiniteTela.in"), "r") as fp:
            ref_deform = fp.read()
        self.assertIn("reference task: no applied strain", ref_deform)

        perturbations = set()
        for task_dir in dfm_dirs[1:]:
            self.assertTrue(os.path.isfile(os.path.join(task_dir, "POSCAR")))
            self.assertTrue(os.path.isfile(os.path.join(task_dir, "FiniteTela.json")))
            self.assertTrue(os.path.isfile(os.path.join(task_dir, "strain.json")))
            self.assertTrue(os.path.isfile(os.path.join(task_dir, "variable_FiniteTela.in")))
            self.assertTrue(os.path.isfile(os.path.join(task_dir, "deform_FiniteTela.in")))

            meta = loadfn(os.path.join(task_dir, "FiniteTela.json"))
            perturbations.add((meta["strain_label"], meta["strain_value"]))

        self.assertEqual(
            {
                ("xx", -0.01),
                ("xx", 0.01),
                ("yz", -0.01),
                ("yz", 0.01),
            },
            perturbations,
        )

        with open(os.path.join(dfm_dirs[-1], "deform_FiniteTela.in"), "r") as fp:
            shear_deform = fp.read()
        self.assertIn("change_box all yz delta ${tilt_delta}", shear_deform)

    def test_average_stress(self):
        task_dir = os.path.join(self.target_path, "task.avg")
        os.makedirs(task_dir, exist_ok=True)

        with open(os.path.join(task_dir, "average_stress.txt"), "w") as fp:
            fp.write("# step pxx pyy pzz pxy pxz pyz\n")
            fp.write("0 1000 2000 3000 400 500 600\n")
            fp.write("1 3000 4000 5000 800 900 1000\n")

        stress = self.finite._average_stress(task_dir)
        self.assertEqual(
            [
                [2.0, 0.6, 0.7],
                [0.6, 3.0, 0.8],
                [0.7, 0.8, 4.0],
            ],
            stress,
        )

    def test_forward_common_files(self):
        fc_files = [
            "in.lammps",
            "variable_FiniteTela.in",
            "deform_FiniteTela.in",
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
