import unittest
from MathsCatGen import maths_catgen

class TestMathsCatGen(unittest.TestCase):
    def test_is_ground_truth_correct(self):
        self.assertTrue(maths_catgen.is_ground_truth_correct("-14338", "-14338"))
        self.assertTrue(maths_catgen.is_ground_truth_correct("-14338 ", "-14338"))
        self.assertTrue(maths_catgen.is_ground_truth_correct("-14338.", "-14338"))
        self.assertTrue(maths_catgen.is_ground_truth_correct("-14,338", "-14338"))
        self.assertTrue(maths_catgen.is_ground_truth_correct("-14,338 ", "-14338"))
        self.assertTrue(maths_catgen.is_ground_truth_correct("-14,338.", "-14338"))
        self.assertFalse(maths_catgen.is_ground_truth_correct("-14339", "-14338"))
        self.assertTrue(maths_catgen.is_ground_truth_correct("boxed{-14338}", "-14338"))
        self.assertTrue(maths_catgen.is_ground_truth_correct("**-14338**", "-14338"))
        self.assertTrue(maths_catgen.is_ground_truth_correct("blah blah-14338 blah blah", "-14338"))
        self.assertTrue(maths_catgen.is_ground_truth_correct("135,702,468 - 269,485,731 = **-133,783,263**", "-133783263"))
        self.assertTrue(maths_catgen.is_ground_truth_correct("12123 - 12312 = -14338", "-14338"))
        self.assertFalse(maths_catgen.is_ground_truth_correct("2", "12"))
        

    def test_calculate_ground_truth(self):
        self.assertEqual(maths_catgen.calculate_ground_truth(2, 3, maths_catgen.MIN_TASK), "2")
        self.assertEqual(maths_catgen.calculate_ground_truth(2, 3, maths_catgen.MAX_TASK), "3")
        self.assertEqual(maths_catgen.calculate_ground_truth(2, 3, maths_catgen.SUM_TASK), "5")
        self.assertEqual(maths_catgen.calculate_ground_truth(2, 3, maths_catgen.DIFF_TASK), "1")
        self.assertEqual(maths_catgen.calculate_ground_truth(2, 3, maths_catgen.PROD_TASK), "6")
        self.assertEqual(maths_catgen.calculate_ground_truth(2, 3, maths_catgen.AVG_TASK), "2.5")
        self.assertEqual(maths_catgen.calculate_ground_truth(2, 20, maths_catgen.EXP_TASK), "1048576")
        self.assertEqual(maths_catgen.calculate_ground_truth(2, 3, maths_catgen.EXP_TASK), "8")

    def test_generate_number_pairs(self):
        pairs = maths_catgen.generate_number_pairs(10, min_val=1, max_val=5, include_negatives=False, seed=1)
        self.assertEqual(len(pairs), 10)

    def test_get_maths_tasks(self):
        tasks = maths_catgen.get_maths_tasks()
        self.assertIn(maths_catgen.MIN_TASK, tasks)
        self.assertIn(maths_catgen.MAX_TASK, tasks)
        self.assertIn(maths_catgen.AVG_TASK, tasks)
        self.assertIn(maths_catgen.SUM_TASK, tasks)
        self.assertIn(maths_catgen.DIFF_TASK, tasks)
        self.assertIn(maths_catgen.PROD_TASK, tasks)
        self.assertIn(maths_catgen.EXP_TASK, tasks)

    def test_get_prompt_template(self):
        template = maths_catgen.get_prompt_template()
        self.assertIn("{x}", template)
        self.assertIn("{y}", template)
        self.assertIn("{task}", template)

    def test_generate_synthetic_data(self):
        tasks = maths_catgen.get_maths_tasks()       
        prompt_template = maths_catgen.get_prompt_template() 
        maths_catgen.generate_synthetic_data(tasks, prompt_template, 4)

    def test_generate_synthetic_matrix(self):
        prompt_template = maths_catgen.get_prompt_template()   
        maths_catgen.generate_synthetic_matrix(prompt_template, 4, 5)
        maths_catgen.generate_synthetic_matrix(prompt_template, 4, 6)
        maths_catgen.generate_synthetic_matrix(prompt_template, 4, 7)
        maths_catgen.generate_synthetic_matrix(prompt_template, 4, 8)

if __name__ == "__main__":
    unittest.main()
