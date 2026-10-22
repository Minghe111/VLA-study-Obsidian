"""Visualize frozen physical records. NO new physics simulation is performed.

All set_pose calls below replay measured trace states. These are NOT evidence
of task success; success evidence is the separate immutable PhysX CPU receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sapien
from transforms3d.quaternions import mat2quat

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"physics_visualization"
FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def material(cfg):
    m=sapien.render.RenderMaterial()
    for key in ("base_color","transmission","ior","roughness","transmission_roughness","metallic"):
        if key in cfg:setattr(m,key,cfg[key])
    return m


def look_at(camera,eye,target):
    eye,target=np.asarray(eye,float),np.asarray(target,float)
    forward=target-eye;forward/=np.linalg.norm(forward)
    left=np.cross([0,0,1.],forward);left/=np.linalg.norm(left)
    up=np.cross(forward,left)
    camera.entity.set_pose(sapien.Pose(eye,mat2quat(np.column_stack((forward,left,up)))))


def aspose(row):
    return sapien.Pose(row[:3],row[3:])


def main(video=True):
    OUT.mkdir(exist_ok=True)
    trace_path=ROOT/"physics_final_seed0_trace.json"
    result_path=ROOT/"physics_final_seed0.json"
    trace=json.loads(trace_path.read_text())
    result=json.loads(result_path.read_text())
    source_hashes=result["source_sha256"]
    assert result["passed"] and all(sha(ROOT/name)==value for name,value in source_hashes.items())
    assert len(trace)==730 and all(len(x["tablets"])==12 for x in trace)
    spec=json.loads((ROOT/"physics_spec.json").read_text())
    assets={x["id"]:x for x in json.loads((ROOT/"asset_interface.json").read_text())["assets"]}
    sapien.render.set_global_config(max_num_materials=50000,max_num_textures=50000)
    sapien.render.set_camera_shader_dir("rt")
    sapien.render.set_ray_tracing_samples_per_pixel(64)
    sapien.render.set_ray_tracing_path_depth(14)
    sapien.render.set_ray_tracing_denoiser("none")
    engine=sapien.Engine();renderer=sapien.SapienRenderer();engine.set_renderer(renderer)
    scene=engine.create_scene(sapien.SceneConfig())
    scene.set_ambient_light([.32,.34,.38])
    scene.add_directional_light([-.3,-.5,-1],[1.4,1.37,1.3],shadow=True)
    scene.add_directional_light([.5,.3,-1],[.8,.85,1.0],shadow=False)
    scene.add_point_light([0,.4,.7],[.7,.7,.7])

    # All entities are static visuals moved only for record playback.
    def static_mesh(name,groups,override=None):
        b=scene.create_actor_builder()
        for group in groups:
            cfg=dict(group["material"])
            if override:cfg.update(override)
            b.add_visual_from_file(group["filename"],scale=[1,1,1],material=material(cfg))
        return b
    b=static_mesh("body",assets["reagent_bottle_body"]["visual_groups"],
        {"base_color":[.96,.90,.72,1],"transmission":.97,"roughness":.035,"transmission_roughness":.008})
    body=b.build_static("trace_bottle_body")
    b=static_mesh("cap",assets["reagent_bottle_cap"]["visual_groups"])
    # A clearly declared visual index mark makes a rotationally symmetric cap
    # readable; it has no collision and does not modify the source OBJ.
    b.add_box_visual(sapien.Pose([.013,0,.00225]),half_size=[.013,.002,.00018],
        material=material({"base_color":[.04,.27,.52,1],"roughness":.5}))
    cap=b.build_static("trace_cap_with_visual_index_mark")
    b=static_mesh("dish",assets["petri_dish"]["visual_groups"])
    b.initial_pose=sapien.Pose(spec["fixture"]["receiver_position_m"])
    dish=b.build_static("trace_imported_petri_dish")
    tablets={}
    for i in range(12):
        b=scene.create_actor_builder()
        color=[.93,.29,.10,1] if i%2==0 else [.96,.81,.37,1]
        b.add_capsule_visual(radius=spec["tablets"]["radius_m"],half_length=spec["tablets"]["half_length_m"],
            material=material({"base_color":color,"roughness":.32}))
        tablets[f"tablet_{i:02d}"]=b.build_static(f"trace_tablet_{i:02d}")
    b=scene.create_actor_builder()
    b.add_box_visual(half_size=[1,1,.012],material=material({"base_color":[.70,.75,.77,1],"roughness":.62}))
    b.initial_pose=sapien.Pose([0,0,-.012])
    b.build_static("studio_worktop")
    # Sparse visual metric grid provides scale and improves glass readability.
    for axis in (0,1):
        for i in range(-5,6):
            b=scene.create_actor_builder()
            half=[.24,.00015,.00002] if axis==0 else [.00015,.24,.00002]
            pos=[0,.05*i,.00004] if axis==0 else [.05*i,0,.00004]
            b.add_box_visual(half_size=half,material=material({"base_color":[.53,.60,.63,1],"roughness":.9}))
            b.initial_pose=sapien.Pose(pos);b.build_static(f"visual_grid_{axis}_{i}")
    w,h=1280,960
    camera=scene.add_camera("physical_trace_overview",w,h,math.radians(31),.001,5)
    look_at(camera,[.31,.40,.29],[.035,0,.120])
    inset=scene.add_camera("physical_trace_closeup",480,330,math.radians(39),.001,5)
    font=ImageFont.truetype(FONT,29)
    small=ImageFont.truetype(FONT,21)
    tiny=ImageFont.truetype(FONT,17)
    phase_titles={
        "upright_settle":"CLOSED BOTTLE / 12 DYNAMIC TABLETS",
        "closed_invert":"CLOSED BOTTLE / INVERSION TEST",
        "closed_inverted_hold":"CLOSED BOTTLE / NO LEAK",
        "return_upright":"RETURN TO UPRIGHT",
        "upright_resettle":"UPRIGHT / TABLETS RESETTLE BY GRAVITY",
        "unscrew_guide":"UNSCREW / MEASURED HELICAL GUIDE",
        "unscrew_hold":"CAP CLEARS GUIDE / RELEASE CHECK",
        "cap_fixture_remove":"DETACHED CAP / FIXTURE REMOVAL",
        "pour_fixture_tilt":"TILT / GRAVITY-DRIVEN POUR",
        "pour_settle":"12 TABLETS IN IMPORTED PETRI DISH",
    }
    rendering_records=[]
    def apply(row):
        body.set_pose(aspose(row["body_pose"]))
        cap.set_pose(aspose(row["cap_pose"]))
        for item in row["tablets"]:tablets[item["name"]].set_pose(aspose(item["pose"]))
        if row["phase"]=="pour_settle":
            center=np.array(spec["fixture"]["receiver_position_m"])+[0,0,.005]
            look_at(inset,center+[.09,.11,.10],center)
        elif row["phase"]=="pour_fixture_tilt":
            center=np.array([.015,0,.075])
            look_at(inset,center+[.12,.14,.11],center)
        else:
            # Follow the actual cap center until removal, not a desired pose.
            center=np.asarray(row["cap_pose"][:3])+[0,0,-.004]
            look_at(inset,center+[.09,.115,.065],center)
    def picture(row):
        apply(row);scene.update_render();camera.take_picture();inset.take_picture()
        frame=Image.fromarray((np.clip(camera.get_picture("Color")[...,:3],0,1)*255).astype(np.uint8))
        close=Image.fromarray((np.clip(inset.get_picture("Color")[...,:3],0,1)*255).astype(np.uint8))
        draw=ImageDraw.Draw(frame)
        draw.rectangle((0,0,w,95),fill="#132c3b")
        draw.text((23,12),"FIXTURE-DRIVEN PHYSICAL TRACE  |  REPLAY",fill="#f2f6f8",font=font)
        phase_title=phase_titles[row["phase"]]
        if row["phase"]=="pour_settle" and any(item["pose"][2]>.020 for item in row["tablets"]):
            phase_title="GRAVITY SETTLING / TABLETS STILL IN MOTION"
        draw.text((23,55),phase_title,fill="#a8d9dc",font=small)
        # Insets are drawn from the same recorded body, cap and tablet states.
        x,y=778,584
        draw.rounded_rectangle((x-4,y-28,x+484,y+334),radius=8,fill="#f4f6f7",outline="#425f6d",width=2)
        draw.text((x+8,y-25),"ACTUAL RECORDED POSES / CLOSE-UP",fill="#274956",font=tiny)
        frame.paste(close,(x,y))
        draw=ImageDraw.Draw(frame)
        draw.rectangle((0,923,w,h),fill="#eef2f4")
        st=row["cap_state"]
        if st["released"]:
            info=f"t={row['time_s']:05.2f}s   step {row['step']:05d}   cap DETACHED / screw guide disabled"
        else:
            info=f"t={row['time_s']:05.2f}s   step {row['step']:05d}   cap {st['rotation_rad']/(2*math.pi):.2f} turns   lift {st['lift_m']*1000:.2f}mm"
        draw.text((18,932),info,fill="#243f4d",font=tiny)
        draw.text((18,886),"No robot control. No new physics steps. Blue mark: visual orientation aid.",fill="#2d4955",font=tiny)
        return frame
    checkpoints=[
        ("01_closed_upright",.8,"CLOSED / 12 TABLETS"),
        ("02_closed_inverted",3.3,"INVERTED / NO LEAK"),
        ("03_unscrew_midway",7.2,"SCREW GUIDE / ROTATION + LIFT"),
        ("04_cap_removed",10.,"CAP REMOVED / OPEN MOUTH"),
        ("05_pour_in_progress",11.4,"TILT / TABLETS IN FLIGHT"),
        ("06_tablets_in_dish",14.6,"12 TABLETS IN PETRI DISH"),
    ]
    stage_files=[]
    for name,t,label in checkpoints:
        row=min(trace,key=lambda row:abs(row["time_s"]-t))
        frame=picture(row);path=OUT/(name+".png");frame.save(path)
        stage_files.append(path)
        rendering_records.append({"path":str(path),"sha256":sha(path),"time_s":row["time_s"],
            "step":row["step"],"phase":row["phase"],"cap_state":row["cap_state"],"pixel_std":float(np.asarray(frame).std())})
        print("STILL",name,flush=True)
    thumb_w,thumb_h=640,480
    sheet=Image.new("RGB",(1320,1664),"#eaf0f3")
    draw=ImageDraw.Draw(sheet)
    draw.text((20,16),"PHYSICAL ASSET VERIFICATION  |  RECORDED TRACE",font=ImageFont.truetype(FONT,26),fill="#183d4e")
    draw.text((20,51),"D6 screw-guide approximation + gravity-driven tablets; fixture driven, no robot policy.",font=tiny,fill="#476270")
    for i,(path,(_,_,label)) in enumerate(zip(stage_files,checkpoints)):
        x=12+(i%2)*652;y=90+(i//2)*520
        sheet.paste(Image.open(path).resize((thumb_w,thumb_h),Image.Resampling.LANCZOS),(x,y))
        draw.text((x+6,y+486),label,font=ImageFont.truetype(FONT,19),fill="#294955")
    sheet_path=OUT/"physics_six_stage_contact_sheet.png"
    sheet.save(sheet_path)
    film=None
    if video:
        fps=15
        times=np.arange(trace[0]["time_s"],trace[-1]["time_s"],1/fps)
        film=OUT/"fixture_driven_physical_trace.mp4"
        import os
        os.environ["IMAGEIO_FFMPEG_EXE"]=imageio_ffmpeg.get_ffmpeg_exe()
        writer=imageio.get_writer(str(film),fps=fps,codec="libx264",quality=8,
            macro_block_size=16,ffmpeg_log_level="error",pixelformat="yuv420p")
        try:
            for i,t in enumerate(times):
                row=min(trace,key=lambda row:abs(row["time_s"]-t))
                writer.append_data(np.asarray(picture(row)))
                if i%20==0:print("VIDEO",i,"/",len(times),flush=True)
        finally:
            writer.close()
        # Video read-back proves usable encoded pixels and the frame clock.
        reader=imageio.get_reader(str(film));metadata=reader.get_meta_data()
        readback_frames=reader.count_frames()
        readback_shape=reader.get_data(0).shape;reader.close()
        assert readback_frames==len(times) and readback_shape[:2]==(h,w)
        print("VIDEO_DONE",readback_frames,flush=True)
    else:
        readback_frames=0;metadata={}
    receipt={"verification_date":"2026-10-22","mode":"measured trace playback only",
        "new_physics_steps":0,"hardware_io":False,"robot_policy":False,
        "input_trace":str(trace_path),"input_trace_sha256":sha(trace_path),
        "input_physics_report":str(result_path),"input_physics_report_sha256":sha(result_path),
        "physics_source_hashes_match":True,"source_script_sha256":sha(__file__),
        "renderer":"SAPIEN 3 Vulkan ray tracing, 64 spp, path depth 14, no denoiser",
        "video":None if film is None else {"path":str(film),"sha256":sha(film),"fps":15,"frames":readback_frames,
            "width":w,"height":h,"readback_passed":True,"duration_s":readback_frames/15},
        "stages":rendering_records,"contact_sheet":str(sheet_path),"contact_sheet_sha256":sha(sheet_path),
        "source_meshes":[{"id":key,"path":group["filename"],"sha256":sha(group["filename"])} for key in ("reagent_bottle_body","reagent_bottle_cap","petri_dish") for group in assets[key]["visual_groups"]],
        "visual_only_overrides":{"bottle_material":{"base_color":[.96,.90,.72,1],"transmission":.97,"roughness":.035},
            "reason":"reduce absorption to reveal independently dynamic tablets; source OBJ unchanged",
            "cap_blue_index_mark":"extra visual primitive, no collision; demonstrates measured twist",
            "table_grid_spacing_m":.05},
        "limitations":["Measured physical records are replayed by assigning visual actor poses; this is not a new dynamics run",
            "The underlying run used declared test fixtures, not robot manipulation", "Guide is a helical constraint surrogate, not real thread contact"]}
    (ROOT/"physics_render_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
    print("DONE",sheet_path,film,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stills-only",action="store_true")
    args=parser.parse_args();main(not args.stills_only)
