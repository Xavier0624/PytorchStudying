import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os
from modelscope import snapshot_download

# 使用的 Stable Diffusion 模型
model_dir = snapshot_download('AI-ModelScope/stable-diffusion-2-1')

# 输入照片路径，请替换为你自己的图片
INPUT_IMAGE = "my_photo.jpg"

# 输出路径
OUTPUT_VANGOGH = "output_van_gogh.jpg"
OUTPUT_UKIYOE = "output_ukiyoe.jpg"

# 图像处理尺寸（与模型训练尺寸一致效果最好）
IMAGE_SIZE = (512, 512)

# 去噪强度：0.0 完全保留原图，1.0 几乎完全重绘
STRENGTH_VANGOGH = 0.7
STRENGTH_UKIYOE = 0.7

# 文本引导强度（越大越严格遵循提示词）
GUIDANCE_SCALE = 7.5

# 随机种子
SEED = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_dir,
    torch_dtype=dtype,
)
pipe = pipe.to(device)

# ---------- 工具函数 ----------
def load_image(path, size=IMAGE_SIZE):
    """加载并预处理图像"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到输入图像: {path}")
    img = Image.open(path).convert("RGB").resize(size)
    return img

def generate_stylized(pipe, image, prompt, strength, guidance_scale=7.5, seed=42):
    """执行 img2img 风格化生成"""
    generator = torch.Generator(device).manual_seed(seed)
    with torch.autocast(device):
        result = pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
    return result

# ---------- 风格提示词定义 ----------
# 风格1：梵高后印象派
STYLE1_PROMPT = (
    "A portrait in the style of Vincent Van Gogh, oil painting, "
    "thick expressive brushstrokes, swirling colors, vibrant and emotional"
)

# 风格2：日本浮世绘
STYLE2_PROMPT = (
    "Japanese ukiyo-e woodblock print, Hokusai style, "
    "flat colors, clean outlines, asymmetric composition, traditional Japanese aesthetic"
)

if __name__ == "__main__":
    content_img = load_image(INPUT_IMAGE)

    img_van_gogh = generate_stylized(
        pipe, content_img,
        prompt=STYLE1_PROMPT,
        strength=STRENGTH_VANGOGH,
        guidance_scale=GUIDANCE_SCALE,
        seed=SEED
    )
    img_van_gogh.save(OUTPUT_VANGOGH)

    img_ukiyoe = generate_stylized(
        pipe, content_img,
        prompt=STYLE2_PROMPT,
        strength=STRENGTH_UKIYOE,
        guidance_scale=GUIDANCE_SCALE,
        seed=SEED
    )
    img_ukiyoe.save(OUTPUT_UKIYOE)


