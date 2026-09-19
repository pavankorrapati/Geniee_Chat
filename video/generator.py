# from __future__ import annotations

# import hashlib
# import math
# import re
# import subprocess
# from pathlib import Path
# from typing import Any

# from PIL import Image, ImageDraw, ImageFont
# import imageio_ffmpeg


# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# VIDEO_DIR = PROJECT_ROOT / "data" / "generated_videos"
# VIDEO_DIR.mkdir(parents=True, exist_ok=True)


# def _font(size: int, bold: bool = False):
#     candidates = []
#     windows = Path("C:/Windows/Fonts")
#     if windows.exists():
#         candidates += [
#             windows / ("arialbd.ttf" if bold else "arial.ttf"),
#             windows / ("segoeuib.ttf" if bold else "segoeui.ttf"),
#         ]
#     for path in candidates:
#         if path.exists():
#             return ImageFont.truetype(str(path), size=size)
#     return ImageFont.load_default()


# def parse_prompt(prompt: str) -> dict[str, Any]:
#     """Small rule-based prompt parser; no LLM or cloud service is used."""
#     text = prompt.strip()
#     lower = text.lower()

#     duration_match = re.search(r"\b(\d{1,2})\s*(?:seconds?|secs?|s)\b", lower)
#     duration = max(2, min(int(duration_match.group(1)), 20)) if duration_match else 5

#     fps_match = re.search(r"\b(\d{1,2})\s*fps\b", lower)
#     fps = max(12, min(int(fps_match.group(1)), 30)) if fps_match else 24

#     size_match = re.search(r"\b(640x360|854x480|1280x720)\b", lower)
#     size = size_match.group(1) if size_match else "640x360"
#     width, height = map(int, size.split("x"))

#     if any(k in lower for k in ("car", "vehicle", "road")):
#         scene = "road"
#     elif "sunset" in lower or "sun rise" in lower or "sunrise" in lower:
#         scene = "sunset"
#     elif any(k in lower for k in ("space", "galaxy", "planet", "stars")):
#         scene = "space"
#     elif any(k in lower for k in ("ocean", "sea", "beach", "waves")):
#         scene = "ocean"
#     elif any(k in lower for k in ("forest", "tree", "woods")):
#         scene = "forest"
#     elif any(k in lower for k in ("city", "building", "street")):
#         scene = "city"
#     else:
#         scene = "abstract"

#     # Remove technical controls from the displayed title.
#     title = re.sub(r"\b\d{1,2}\s*(?:seconds?|secs?|s)\b", "", text, flags=re.I)
#     title = re.sub(r"\b\d{1,2}\s*fps\b", "", title, flags=re.I)
#     title = re.sub(r"\b(?:640x360|854x480|1280x720)\b", "", title, flags=re.I)
#     title = re.sub(r"\s+", " ", title).strip(" .,-")
#     if not title:
#         title = "Geniee Video"

#     return {
#         "prompt": text,
#         "title": title[:120],
#         "scene": scene,
#         "duration": duration,
#         "fps": fps,
#         "width": width,
#         "height": height,
#     }


# def _gradient(draw: ImageDraw.ImageDraw, width: int, height: int, top, bottom):
#     for y in range(height):
#         t = y / max(1, height - 1)
#         c = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
#         draw.line((0, y, width, y), fill=c)


# def _draw_scene(draw, spec, t, frame, seed):
#     w, h = spec["width"], spec["height"]
#     scene = spec["scene"]

#     if scene == "sunset":
#         _gradient(draw, w, h, (18, 25, 67), (244, 116, 73))
#         sun_x = int(w * (0.18 + 0.64 * t))
#         sun_y = int(h * (0.52 - 0.20 * math.sin(math.pi * t)))
#         r = max(18, h // 12)
#         draw.ellipse((sun_x-r, sun_y-r, sun_x+r, sun_y+r), fill=(255, 220, 112))
#         horizon = int(h * .70)
#         draw.rectangle((0, horizon, w, h), fill=(26, 28, 48))
#         for x in range(-w, w * 2, 70):
#             draw.polygon([(x, horizon), (x+45, horizon), (x+22, horizon-55), (x+5, horizon-20)], fill=(20,22,38))

#     elif scene == "space":
#         _gradient(draw, w, h, (4, 6, 25), (20, 8, 45))
#         rng = seed
#         for i in range(90):
#             x = (i * 83 + rng * 17) % w
#             y = (i * 47 + rng * 11) % h
#             twinkle = 1 + int(2 * (0.5 + 0.5 * math.sin(t * 8 + i)))
#             draw.ellipse((x, y, x+twinkle, y+twinkle), fill=(235, 240, 255))
#         cx = int(w * (0.30 + 0.35 * t))
#         cy = int(h * .50)
#         r = int(h * .16)
#         draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(92, 115, 205))
#         draw.ellipse((cx-r//2, cy-r//2, cx+r//3, cy+r//3), fill=(70, 87, 160))

#     elif scene == "ocean":
#         _gradient(draw, w, h, (28, 94, 155), (10, 35, 85))
#         horizon = int(h * .52)
#         draw.rectangle((0, horizon, w, h), fill=(11, 74, 125))
#         for i in range(12):
#             y = horizon + i * 18
#             offset = int(math.sin(t * 5 + i) * 20)
#             for x in range(-30, w + 30, 90):
#                 draw.arc((x+offset, y-12, x+60+offset, y+12), 180, 350, fill=(80, 175, 215), width=2)
#         draw.ellipse((int(w*.76), int(h*.12), int(w*.86), int(h*.22)), fill=(255, 229, 150))

#     elif scene == "forest":
#         _gradient(draw, w, h, (25, 72, 55), (8, 25, 20))
#         ground = int(h * .70)
#         draw.rectangle((0, ground, w, h), fill=(12, 42, 25))
#         for i in range(12):
#             x = int((i / 11) * w)
#             top = 90 + (i * 17) % 90
#             draw.rectangle((x-8, top, x+8, ground), fill=(70, 45, 28))
#             draw.polygon([(x, top-80), (x-55, top+20), (x+55, top+20)], fill=(18, 74, 42))
#             draw.polygon([(x, top-120), (x-42, top-25), (x+42, top-25)], fill=(20, 88, 47))

#     elif scene == "city":
#         _gradient(draw, w, h, (12, 17, 40), (52, 35, 65))
#         base = int(h*.78)
#         draw.rectangle((0, base, w, h), fill=(22, 22, 27))
#         for i in range(10):
#             bw = 45 + (i*13)%55
#             bh = 80 + (i*37)%(h//2)
#             x = i*(w//9)-10
#             y = base-bh
#             draw.rectangle((x,y,x+bw,base), fill=(28, 34, 48))
#             for wy in range(y+12, base-8, 20):
#                 draw.rectangle((x+8,wy,x+14,wy+7), fill=(236, 190, 88))
#                 if x+25 < w:
#                     draw.rectangle((x+25,wy,x+31,wy+7), fill=(236, 190, 88))
#         draw.line((0, base, w, base), fill=(90, 90, 100), width=2)

#     elif scene == "road":
#         _gradient(draw, w, h, (85, 145, 190), (30, 55, 75))
#         horizon = int(h*.52)
#         draw.polygon([(w*.42,horizon),(w*.58,horizon),(w,h),(0,h)], fill=(45,45,48))
#         for k in range(7):
#             y = horizon + int((h-horizon)*(k/7)**1.6)
#             half = int(3 + (y-horizon)*.05)
#             cx = w//2
#             draw.rectangle((cx-half,y,cx+half,y+int(12+half)), fill=(235, 220, 120))
#         car_x = int(w*.50 + math.sin(t*math.pi*2)*w*.22)
#         car_y = int(h*.70)
#         draw.rounded_rectangle((car_x-45,car_y-18,car_x+45,car_y+18), radius=8, fill=(180,35,35))
#         draw.polygon([(car_x-25,car_y-18),(car_x-10,car_y-38),(car_x+18,car_y-38),(car_x+30,car_y-18)], fill=(120,160,190))
#         draw.ellipse((car_x-33,car_y+8,car_x-18,car_y+23), fill=(20,20,20))
#         draw.ellipse((car_x+18,car_y+8,car_x+33,car_y+23), fill=(20,20,20))

#     else:
#         _gradient(draw, w, h, (38, 48, 105), (112, 46, 110))
#         cx = int(w * (.5 + .25 * math.sin(2*math.pi*t)))
#         cy = int(h * (.5 + .20 * math.cos(2*math.pi*t)))
#         r = int(min(w,h) * (.12 + .06 * math.sin(4*math.pi*t)))
#         draw.ellipse((cx-r,cy-r,cx+r,cy+r), fill=(130, 220, 245))
#         for i in range(8):
#             ang = 2*math.pi*i/8 + t*2
#             x = int(cx + math.cos(ang)*(r+45))
#             y = int(cy + math.sin(ang)*(r+45))
#             draw.ellipse((x-7,y-7,x+7,y+7), fill=(240,180,90))


# def create_video(prompt: str, output_path: Path | None = None) -> dict[str, Any]:
#     spec = parse_prompt(prompt)
#     if output_path is None:
#         digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
#         output_path = VIDEO_DIR / f"geniee_{digest}.mp4"
#     output_path = Path(output_path)
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
#     cmd = [
#         ffmpeg, "-y",
#         "-f", "rawvideo",
#         "-vcodec", "rawvideo",
#         "-pix_fmt", "rgb24",
#         "-s", f"{spec['width']}x{spec['height']}",
#         "-r", str(spec["fps"]),
#         "-i", "-",
#         "-an",
#         "-c:v", "libx264",
#         "-preset", "veryfast",
#         "-pix_fmt", "yuv420p",
#         str(output_path),
#     ]

#     process = subprocess.Popen(
#         cmd,
#         stdin=subprocess.PIPE,
#         stdout=subprocess.DEVNULL,
#         stderr=subprocess.PIPE,
#     )

#     try:
#         total = spec["duration"] * spec["fps"]
#         seed = int(hashlib.sha1(prompt.encode()).hexdigest()[:6], 16) % 997
#         title_font = _font(max(18, spec["height"] // 16), bold=True)
#         small_font = _font(max(11, spec["height"] // 32))

#         for frame in range(total):
#             t = frame / max(1, total - 1)
#             image = Image.new("RGB", (spec["width"], spec["height"]))
#             draw = ImageDraw.Draw(image)
#             _draw_scene(draw, spec, t, frame, seed)

#             # Subtle title card overlay.
#             title = spec["title"]
#             bbox = draw.textbbox((0, 0), title, font=title_font)
#             tw = bbox[2] - bbox[0]
#             x = (spec["width"] - tw) // 2
#             y = 18
#             draw.rounded_rectangle(
#                 (x-14, y-8, x+tw+14, y+(bbox[3]-bbox[1])+10),
#                 radius=10,
#                 fill=(0, 0, 0, 120),
#             )
#             draw.text((x, y), title, font=title_font, fill=(255,255,255))

#             draw.text(
#                 (12, spec["height"]-22),
#                 f"Geniee • {spec['scene']} • {spec['fps']} FPS",
#                 font=small_font,
#                 fill=(225,225,225),
#             )

#             process.stdin.write(image.tobytes())

#         process.stdin.close()
#         stderr = process.stderr.read().decode("utf-8", errors="replace")
#         returncode = process.wait()
#         if returncode != 0:
#             raise RuntimeError(f"FFmpeg failed: {stderr[-1500:]}")
#     except Exception:
#         try:
#             process.stdin.close()
#         except Exception:
#             pass
#         process.kill()
#         process.wait()
#         raise

#     return {
#         "file": output_path.name,
#         "path": str(output_path),
#         "url": f"/videos/{output_path.name}",
#         "duration": spec["duration"],
#         "fps": spec["fps"],
#         "width": spec["width"],
#         "height": spec["height"],
#         "scene": spec["scene"],
#         "title": spec["title"],
#     }

from __future__ import annotations

import hashlib
import math
import re
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_ROOT / "data" / "generated_videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)


def _font(size: int, bold: bool = False):
    candidates = []
    windows = Path("C:/Windows/Fonts")
    if windows.exists():
        candidates += [
            windows / ("arialbd.ttf" if bold else "arial.ttf"),
            windows / ("segoeuib.ttf" if bold else "segoeui.ttf"),
        ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def parse_prompt(
    prompt: str,
    duration: int | None = None,
    fps: int | None = None,
    resolution: str | None = None,
) -> dict[str, Any]:
    """Parse prompt details, allowing explicit argument overrides from API requests."""
    text = prompt.strip()
    lower = text.lower()

    # Determine duration: explicit API parameter overrides prompt text
    if duration is None:
        duration_match = re.search(r"\b(\d{1,2})\s*(?:seconds?|secs?|s)\b", lower)
        duration = int(duration_match.group(1)) if duration_match else 5
    duration = max(2, min(int(duration), 20))

    # Determine FPS
    if fps is None:
        fps_match = re.search(r"\b(\d{1,2})\s*fps\b", lower)
        fps = int(fps_match.group(1)) if fps_match else 24
    fps = max(12, min(int(fps), 30))

    # Determine resolution/size
    if resolution is None:
        size_match = re.search(r"\b(640x360|854x480|1280x720)\b", lower)
        size = size_match.group(1) if size_match else "640x360"
    else:
        size = resolution

    try:
        width, height = map(int, size.lower().split("x"))
    except ValueError:
        width, height = 640, 360

    if any(k in lower for k in ("car", "vehicle", "road")):
        scene = "road"
    elif "sunset" in lower or "sun rise" in lower or "sunrise" in lower:
        scene = "sunset"
    elif any(k in lower for k in ("space", "galaxy", "planet", "stars")):
        scene = "space"
    elif any(k in lower for k in ("ocean", "sea", "beach", "waves")):
        scene = "ocean"
    elif any(k in lower for k in ("forest", "tree", "woods")):
        scene = "forest"
    elif any(k in lower for k in ("city", "building", "street")):
        scene = "city"
    else:
        scene = "abstract"

    # Remove technical controls from the displayed title
    title = re.sub(r"\b\d{1,2}\s*(?:seconds?|secs?|s)\b", "", text, flags=re.I)
    title = re.sub(r"\b\d{1,2}\s*fps\b", "", title, flags=re.I)
    title = re.sub(r"\b(?:640x360|854x480|1280x720)\b", "", title, flags=re.I)
    title = re.sub(r"\s+", " ", title).strip(" .,-")
    if not title:
        title = "Geniee Video"

    return {
        "prompt": text,
        "title": title[:120],
        "scene": scene,
        "duration": duration,
        "fps": fps,
        "width": width,
        "height": height,
    }


def _gradient(draw: ImageDraw.ImageDraw, width: int, height: int, top, bottom):
    for y in range(height):
        t = y / max(1, height - 1)
        c = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
        draw.line((0, y, width, y), fill=c)


def _draw_scene(draw, spec, t, frame, seed):
    w, h = spec["width"], spec["height"]
    scene = spec["scene"]

    if scene == "sunset":
        _gradient(draw, w, h, (18, 25, 67), (244, 116, 73))
        sun_x = int(w * (0.18 + 0.64 * t))
        sun_y = int(h * (0.52 - 0.20 * math.sin(math.pi * t)))
        r = max(18, h // 12)
        draw.ellipse((sun_x-r, sun_y-r, sun_x+r, sun_y+r), fill=(255, 220, 112))
        horizon = int(h * .70)
        draw.rectangle((0, horizon, w, h), fill=(26, 28, 48))
        for x in range(-w, w * 2, 70):
            draw.polygon([(x, horizon), (x+45, horizon), (x+22, horizon-55), (x+5, horizon-20)], fill=(20,22,38))

    elif scene == "space":
        _gradient(draw, w, h, (4, 6, 25), (20, 8, 45))
        rng = seed
        for i in range(90):
            x = (i * 83 + rng * 17) % w
            y = (i * 47 + rng * 11) % h
            twinkle = 1 + int(2 * (0.5 + 0.5 * math.sin(t * 8 + i)))
            draw.ellipse((x, y, x+twinkle, y+twinkle), fill=(235, 240, 255))
        cx = int(w * (0.30 + 0.35 * t))
        cy = int(h * .50)
        r = int(h * .16)
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(92, 115, 205))
        draw.ellipse((cx-r//2, cy-r//2, cx+r//3, cy+r//3), fill=(70, 87, 160))

    elif scene == "ocean":
        _gradient(draw, w, h, (28, 94, 155), (10, 35, 85))
        horizon = int(h * .52)
        draw.rectangle((0, horizon, w, h), fill=(11, 74, 125))
        for i in range(12):
            y = horizon + i * 18
            offset = int(math.sin(t * 5 + i) * 20)
            for x in range(-30, w + 30, 90):
                draw.arc((x+offset, y-12, x+60+offset, y+12), 180, 350, fill=(80, 175, 215), width=2)
        draw.ellipse((int(w*.76), int(h*.12), int(w*.86), int(h*.22)), fill=(255, 229, 150))

    elif scene == "forest":
        _gradient(draw, w, h, (25, 72, 55), (8, 25, 20))
        ground = int(h * .70)
        draw.rectangle((0, ground, w, h), fill=(12, 42, 25))
        for i in range(12):
            x = int((i / 11) * w)
            top = 90 + (i * 17) % 90
            draw.rectangle((x-8, top, x+8, ground), fill=(70, 45, 28))
            draw.polygon([(x, top-80), (x-55, top+20), (x+55, top+20)], fill=(18, 74, 42))
            draw.polygon([(x, top-120), (x-42, top-25), (x+42, top-25)], fill=(20, 88, 47))

    elif scene == "city":
        _gradient(draw, w, h, (12, 17, 40), (52, 35, 65))
        base = int(h*.78)
        draw.rectangle((0, base, w, h), fill=(22, 22, 27))
        for i in range(10):
            bw = 45 + (i*13)%55
            bh = 80 + (i*37)%(h//2)
            x = i*(w//9)-10
            y = base-bh
            draw.rectangle((x,y,x+bw,base), fill=(28, 34, 48))
            for wy in range(y+12, base-8, 20):
                draw.rectangle((x+8,wy,x+14,wy+7), fill=(236, 190, 88))
                if x+25 < w:
                    draw.rectangle((x+25,wy,x+31,wy+7), fill=(236, 190, 88))
        draw.line((0, base, w, base), fill=(90, 90, 100), width=2)

    elif scene == "road":
        _gradient(draw, w, h, (85, 145, 190), (30, 55, 75))
        horizon = int(h*.52)
        draw.polygon([(w*.42,horizon),(w*.58,horizon),(w,h),(0,h)], fill=(45,45,48))
        for k in range(7):
            y = horizon + int((h-horizon)*(k/7)**1.6)
            half = int(3 + (y-horizon)*.05)
            cx = w//2
            draw.rectangle((cx-half,y,cx+half,y+int(12+half)), fill=(235, 220, 120))
        car_x = int(w*.50 + math.sin(t*math.pi*2)*w*.22)
        car_y = int(h*.70)
        draw.rounded_rectangle((car_x-45,car_y-18,car_x+45,car_y+18), radius=8, fill=(180,35,35))
        draw.polygon([(car_x-25,car_y-18),(car_x-10,car_y-38),(car_x+18,car_y-38),(car_x+30,car_y-18)], fill=(120,160,190))
        draw.ellipse((car_x-33,car_y+8,car_x-18,car_y+23), fill=(20,20,20))
        draw.ellipse((car_x+18,car_y+8,car_x+33,car_y+23), fill=(20,20,20))

    else:
        _gradient(draw, w, h, (38, 48, 105), (112, 46, 110))
        cx = int(w * (.5 + .25 * math.sin(2*math.pi*t)))
        cy = int(h * (.5 + .20 * math.cos(2*math.pi*t)))
        r = int(min(w,h) * (.12 + .06 * math.sin(4*math.pi*t)))
        draw.ellipse((cx-r,cy-r,cx+r,cy+r), fill=(130, 220, 245))
        for i in range(8):
            ang = 2*math.pi*i/8 + t*2
            x = int(cx + math.cos(ang)*(r+45))
            y = int(cy + math.sin(ang)*(r+45))
            draw.ellipse((x-7,y-7,x+7,y+7), fill=(240,180,90))


def generate_video(
    prompt: str,
    output_path: Path | None = None,
    duration: int | None = None,
    fps: int | None = None,
    resolution: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    spec = parse_prompt(prompt, duration=duration, fps=fps, resolution=resolution)

    if output_path is None:
        digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
        output_path = VIDEO_DIR / f"geniee_{digest}.mp4"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{spec['width']}x{spec['height']}",
        "-r", str(spec["fps"]),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        total = spec["duration"] * spec["fps"]
        seed = int(hashlib.sha1(prompt.encode()).hexdigest()[:6], 16) % 997
        title_font = _font(max(18, spec["height"] // 16), bold=True)
        small_font = _font(max(11, spec["height"] // 32))

        for frame in range(total):
            t = frame / max(1, total - 1)
            image = Image.new("RGB", (spec["width"], spec["height"]))
            draw = ImageDraw.Draw(image)
            _draw_scene(draw, spec, t, frame, seed)

            # Subtle title card overlay.
            title = spec["title"]
            bbox = draw.textbbox((0, 0), title, font=title_font)
            tw = bbox[2] - bbox[0]
            x = (spec["width"] - tw) // 2
            y = 18
            draw.rounded_rectangle(
                (x-14, y-8, x+tw+14, y+(bbox[3]-bbox[1])+10),
                radius=10,
                fill=(0, 0, 0, 120),
            )
            draw.text((x, y), title, font=title_font, fill=(255,255,255))

            draw.text(
                (12, spec["height"]-22),
                f"Geniee • {spec['scene']} • {spec['fps']} FPS",
                font=small_font,
                fill=(225,225,225),
            )

            process.stdin.write(image.tobytes())

        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {stderr[-1500:]}")
    except Exception:
        try:
            process.stdin.close()
        except Exception:
            pass
        process.kill()
        process.wait()
        raise

    return {
        "file": output_path.name,
        "path": str(output_path),
        "url": f"/videos/{output_path.name}",
        "duration": spec["duration"],
        "fps": spec["fps"],
        "width": spec["width"],
        "height": spec["height"],
        "scene": spec["scene"],
        "title": spec["title"],
    }