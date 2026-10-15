"""Stdlib CPU-only boundaries for initial support collision classification.
Extracts only the pure predicate; no MPlib, GPU, worker, or simulation imports.
"""
from pathlib import Path
import ast,hashlib,unittest
BASE=Path(__file__).resolve().parent
OLD=BASE/'full_native_carry_planner_v21.py'
NEW=BASE/'full_native_carry_planner_v31.py'
def predicate():
 tree=ast.parse(NEW.read_text(encoding='utf-8'))
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_support_start_pair_only')
 context={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(NEW),'exec'),context)
 return context['_support_start_pair_only']
def pair(a='held_bottle',b='realman_table'):return {'link_name1':a,'link_name2':b,'object_name1':a,'object_name2':b}
class SupportStartBoundaries(unittest.TestCase):
 def test_empty_collision_set_is_eligible_for_remaining_strict_guards(self):
  self.assertTrue(predicate()([]))
 def test_sole_support_pair_both_orderings(self):
  fn=predicate();self.assertTrue(fn([pair()]));self.assertTrue(fn([pair('realman_table','held_bottle')]));self.assertTrue(fn([pair(),pair()]))
 def test_any_foreign_pair_rejected(self):
  fn=predicate()
  for collision in [pair('held_bottle','body_base_link'),pair('l_link4','realman_table'),pair('body_base_link','l_link4'),{}, {'object_name1':'held_bottle','object_name2':'realman_table'}]:
   with self.subTest(collision=collision):
    self.assertFalse(fn([collision]));self.assertFalse(fn([pair(),collision]))
 def test_only_target_function_AST_changed(self):
  old=ast.parse(OLD.read_text(encoding='utf-8'));new=ast.parse(NEW.read_text(encoding='utf-8'))
  oldfunc=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='_support_start_pair_only')
  newfunc=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='_support_start_pair_only')
  self.assertNotEqual(ast.dump(oldfunc,include_attributes=False),ast.dump(newfunc,include_attributes=False))
  for t in (old,new):t.body=[n for n in t.body if not isinstance(n,ast.FunctionDef) or n.name!='_support_start_pair_only']
  self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))
  self.assertEqual(hashlib.sha256(OLD.read_bytes()).hexdigest(),'65b9a084d0ba7941bfa71dd16cb487d721cf136800ab1d370b65b3123daba40f')
 def test_exact_single_source_line_delta(self):
  a=OLD.read_text(encoding='utf-8').splitlines();b=NEW.read_text(encoding='utf-8').splitlines();self.assertEqual(len(a),len(b))
  changes=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
  self.assertEqual(changes,[164]);self.assertEqual(a[164].replace('bool(collisions) and ',''),b[164])

if __name__=='__main__':unittest.main(verbosity=2)
