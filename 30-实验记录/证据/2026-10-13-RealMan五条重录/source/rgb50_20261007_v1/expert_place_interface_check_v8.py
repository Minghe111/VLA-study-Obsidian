"""Run v7 contract checks against strict v8 final-pair implementation."""
from pathlib import Path
import expert_generation_v7 as v7
import expert_generation_v8 as v8

source=Path(__file__).with_name('expert_place_interface_check_v7.py').read_text()
source=source.replace('session=v7.ExpertSession.__new__(v7.ExpertSession)',
                      'session=v8.ExpertSession.__new__(v8.ExpertSession)')
source=source.replace("observed.append((self.dense_calls,self.placement_candidates[0]))",
    "observed.append((self.dense_calls,dict(self.placement_candidates[0],candidate_count=len(self.placement_candidates))))")
source=source.replace("assert observed[1][1]['yaw_delta_deg']==30.",
    "assert observed[1][1]['yaw_delta_deg']==30.\nassert observed[1][1]['candidate_count']==1\nassert session.report['strict_placement_pair']['different_yaw_or_corridor_fallback_allowed'] is False")
source=source.replace("'selected_yaw_retained_for_final_offset':True,",
    "'selected_yaw_retained_for_final_offset':True,'final_pair_candidate_count':observed[1][1]['candidate_count'],'alternative_final_fallback_allowed':False,")
exec(compile(source,str(Path(__file__).with_name('expert_place_interface_check_v7.py')),'exec'))
