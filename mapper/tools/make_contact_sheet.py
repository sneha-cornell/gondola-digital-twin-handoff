from pathlib import Path

from PIL import Image, ImageDraw

ids = [
    "00001", "00002", "00011", "00012", "00013",
    "00024", "00025", "00026", "00033", "00034",
    "00035", "00036", "00037", "00038", "00039",
    "00041", "00043", "00052", "00053", "00058",
]
base = Path("/home/ec2-user/SageMaker/product_detection1/backend/workspace/iphone16-2/images")
out = Path("/home/ec2-user/SageMaker/product_detection1/.tmp/reference_frames.jpg")
out.parent.mkdir(parents=True, exist_ok=True)

thumb_w, thumb_h = 162, 288
label_h = 30
cols = 5
rows = (len(ids) + cols - 1) // cols
sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
draw = ImageDraw.Draw(sheet)

for idx, image_id in enumerate(ids):
    img = Image.open(base / f"{image_id}.jpg").convert("RGB")
    img.thumbnail((thumb_w, thumb_h))
    x = (idx % cols) * thumb_w
    y = (idx // cols) * (thumb_h + label_h)
    draw.text((x + 5, y + 5), image_id, fill=(0, 0, 0))
    sheet.paste(img, (x + (thumb_w - img.width) // 2, y + label_h))

sheet.save(out, quality=92)
print(out)
