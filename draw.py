
from PIL import Image
import io
import base64
import json
from http import HTTPStatus
import requests
from dashscope import ImageSynthesis
import os
from io import BytesIO
from api_key import api_key
from openai import OpenAI
from datetime import datetime





def generate_image_dall(prompt,model):

    url = "https://xiaohumini.site/v1/images/generations"
    headers = {
       'Content-Type': 'application/json',
    'Authorization':api_key("xiaohu")
    }

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024", #生成图像的大小。必须是256x256、512x512或1024x1024for之一dall - e - 2。对于模型来说，必须是1024x1024、1792x1024
        'quality':'standard', # standard  hd
        'user':'natural', # natural vivid
        'response_format':'url' #url   b64_json
    })

    response = requests.request("POST", url, headers=headers, data=payload)
    response_data = json.loads(response.text)
    print(response_data)
    url = response_data['data'][0]['url']

    return url




def generate_image_flux(prompt,model):

    url = "https://xiaohumini.site/v1/images/generations"
    headers = {
       'Content-Type': 'application/json',
    'Authorization':api_key("xiaohu")
    }

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024", #生成图像的大小。必须是256x256、512x512或1024x1024for之一dall - e - 2。对于模型来说，必须是1024x1024、1792x1024
        'quality':'standard', # standard  hd
        'user':'natural', # natural vivid
        'response_format':'url' #url   b64_json
    })

    response = requests.request("POST", url, headers=headers, data=payload)
    response_data = json.loads(response.text)
    print(response_data)
    url = response_data['data'][0]['url']

    return url




def generate_image_MJ(prompt,model):

    url = "https://xiaohumini.site/v1/mj/submit/imagine"
    headers = {
       'Content-Type': 'application/json',
        'Authorization':api_key("xiaohu")
    }

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
    })

    response = requests.request("POST", url, headers=headers, data=payload)
    response_data = json.loads(response.text)

    id = response_data['result']

    url = "https://xiaohumini.site/mj/task/list-by-condition"

    payload = json.dumps({
        "ids": [
            id
        ]
    })
    while 1:
        #sleep(2) #等待两秒
        response = requests.request("POST", url, headers=headers, data=payload)
        response_data = json.loads(response.text)
        print(response_data[0]['progress'])
        if response_data[0]['progress'] =='100%':
            url = response_data[0]['imageUrl']
            return url


def generate_image_Qwen(prompt,model):
    rsp = ImageSynthesis.call(
        api_key=os.getenv("DASHSCOPE_API_KEY", api_key("qwen")),
        model=model,
        prompt=prompt,
        n=1,
        prompt_extend='false',
        size='1024*1024'
    )

    if rsp.status_code == HTTPStatus.OK:
        # 仅输出图片 URL
        for result in rsp.output.results:
            return result.url






def generate_image_minimax(prompt,model):

    url = "https://api.minimax.chat/v1/image_generation"
    payload = json.dumps({
    "model": model, 
    "prompt": prompt,
    "aspect_ratio": "1:1",
    "response_format": "url",
    "n": 1,
    "prompt_optimizer": False
    })
    headers = {
    'Authorization': f'Bearer {api_key("minimax")}',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response_data = json.loads(response.text)

    url = response_data['data']['image_urls'][0]

    return url



def generate_image_GPT(prompt,model):

    url = "https://aihubmix.com/v1/images/generations"
    headers = {
       'Content-Type': 'application/json',
    'Authorization':api_key("aihubmix")
    }

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024", # 1024x1024 (方形), 1536x1024 (3:2 景观), 1024x1536 (2:3 肖像), auto (默认自动比例，不需要显式传入)
        'response_format':'url' #url   b64_json
    })

    response = requests.request("POST", url, headers=headers, data=payload)
    response_data = json.loads(response.text)
    url = response_data['data'][0]['url']

    return url


def generate_image_gpt_image(prompt,model,save_path):

   # url = "https://xiaohumini.site/v1/images/generations"
    url = 'https://aihubmix.com/v1/images/generations'
    headers = {
       'Content-Type': 'application/json',
    'Authorization':api_key("aihubmix")
    }

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024", #生成图像的大小。必须是256x256、512x512或1024x1024for之一dall - e - 2。对于模型来说，必须是1024x1024、1792x1024
        'quality':'medium', # low high medium
        'moderation':'low',

    })

    response = requests.request("POST", url, headers=headers, data=payload)
    image_base64 = response.json()["data"][0]["b64_json"]
    image_bytes = base64.b64decode(image_base64)

    img = Image.open(BytesIO(image_bytes))

    img.save(save_path, "PNG")

    # 获取原始图像尺寸
    original_width, original_height = img.size

    # 计算新的尺寸（宽度和高度都缩小一半）
    new_width = original_width // 2
    new_height = original_height // 2

    # 使用 resize 方法改变图像尺寸
    # Image.BICUBIC 是一个常用的高质量缩放滤波器
    resized_img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)

    # 创建一个 BytesIO 对象来存储压缩后的图像数据
    compressed_image_buffer = BytesIO()

    # 将缩小尺寸后的图像保存到 BytesIO
    # 保存为 PNG 格式
    resized_img.save(compressed_image_buffer, format="PNG")

    # 将 BytesIO 的指针移到开始位置
    compressed_image_buffer.seek(0)

    # 将压缩（尺寸缩小）后的图像数据写入文件
    with open(f'{save_path[:-4]}_resize.png', "wb") as f:
        f.write(compressed_image_buffer.read())

    # 重新读取处理后的文件进行 base64 编码
    with open(f'{save_path[:-4]}_resize.png', "rb") as image_file:
        base64_encoder = base64.b64encode(image_file.read()).decode('utf-8')


    os.remove(f'{save_path[:-4]}_resize.png')

    return base64_encoder





def generate_image_gpt_edits(prompt,model,file_path,save_path):


    image = []
    for input_dir in file_path:
        image.append(open(input_dir, "rb"))

    if not image:
        raise ValueError("No image files provided")

    from openai import OpenAI
    client = OpenAI(
        api_key=api_key("aihubmix") , # 换成你在后台生成的 Key "sk-***"
        base_url="https://aihubmix.com/v1"
    )

    response = client.images.edit(
        model=model,
        image=image[0] if len(file_path)==1 else image,  # 多参考图应使用 [列表，]
        n=1,  # 单次数量
        prompt=prompt,
        size="1024x1024",  # 1024x1024 (square), 1536x1024 (3:2 landscape), 1024x1536 (2:3 portrait), auto (default)

        quality="medium"  # high, medium, low, auto (default)
    )

    image_base64 = response.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    img = Image.open(BytesIO(image_bytes))

    img.save(save_path, "PNG")

    # 获取原始图像尺寸
    original_width, original_height = img.size

    # 计算新的尺寸（宽度和高度都缩小一半）
    new_width = original_width // 2
    new_height = original_height // 2

    # 使用 resize 方法改变图像尺寸
    # Image.BICUBIC 是一个常用的高质量缩放滤波器
    resized_img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)

    # 创建一个 BytesIO 对象来存储压缩后的图像数据
    compressed_image_buffer = BytesIO()

    # 将缩小尺寸后的图像保存到 BytesIO
    # 保存为 PNG 格式
    resized_img.save(compressed_image_buffer, format="PNG")

    # 将 BytesIO 的指针移到开始位置
    compressed_image_buffer.seek(0)

    # 将压缩（尺寸缩小）后的图像数据写入文件
    with open(f'{save_path[:-4]}_resize.png', "wb") as f:
        f.write(compressed_image_buffer.read())

    # 重新读取处理后的文件进行 base64 编码
    with open(f'{save_path[:-4]}_resize.png', "rb") as image_file:
        base64_encoder = base64.b64encode(image_file.read()).decode('utf-8')

    os.remove(f'{save_path[:-4]}_resize.png')

    return base64_encoder

def url_2_base64(url,save_path):

    # 下载图片内容
    image_response = requests.get(url)
    image_response.raise_for_status() # 检查是否成功下载

    # 从响应内容中读取图片数据
    image_data = io.BytesIO(image_response.content)

    # 使用 Pillow 打开图片
    img = Image.open(image_data)


    img.save(save_path, "PNG")

    # 将图片压缩2倍 （等比例缩放）
    width, height = img.size
    new_width = width // 4
    new_height = height // 4
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)  # 使用高质量的RESAMPLING算法

    # 将图片保存为 PNG 格式
    img.save(f'{save_path[:-4]}_resize.png', "PNG")

    # 将图片转化为 base64 编码
    with open(save_path, "rb") as image_file:
     base64_encoder = base64.b64encode(image_file.read()).decode('utf-8')

    os.remove(f'{save_path[:-4]}_resize.png')
    return base64_encoder,url




def generate_image_ideogram(prompt,model):
    data = {
        "prompt": prompt,
        "rendering_speed": "DEFAULT",  # TURBO, DEFAULT, QUALITY
        "num_images": "1",
        "aspect_ratio": "1x1",
        "magic_prompt": "OFF",
        "style_type": "AUTO",
       # "negative_prompt": "blurry, watermark"
    }

    # Content-Type 为 multipart/form-data
    files = {}
    for key, value in data.items():
        files[key] = (None, str(value))  # 将每个数据字段作为表单字段发送

    response = requests.post(
        "https://aihubmix.com/ideogram/v1/ideogram-v3/generate",
        headers={
            "Api-Key": api_key("aihubmix")  # 换成你在 AiHubMix 生成的密钥
        },
        files=files
    )

    response_json = response.json()
    url = response_json['data'][0]['url']

    return url



def generate_image_ideogram_remix(prompt,model,file_path):
    data = {
        "prompt": prompt,
        "image_weight": "50",
        "rendering_speed": "DEFAULT", # TURBO, DEFAULT, QUALITY
        "num_images": 1,
        "seed": 1,
        "aspect_ratio": "1x1",
        "magic_prompt": "OFF",
        "style_type": "AUTO",
      #  "negative_prompt": "blurry, bad anatomy, watermark",
    }

    image = []
    for input_dir in file_path:
        image.append(input_dir)

    if not image:
        raise ValueError("No image files provided")


    # 准备文件上传
    with open(image[0], "rb") as image_file:

        files = {
                "image": image_file,
            }

        response = requests.post(
            "https://aihubmix.com/ideogram/v1/ideogram-v3/remix",
            headers={
                "Api-Key": api_key("aihubmix") # 换成你在 AiHubMix 生成的密钥
            },
            data=data,
            files=files
        )


    response_json = response.json()
    url = response_json['data'][0]['url']

    return url




def generate_image_gemini(prompt,model,save_path):
    client = OpenAI(
        api_key=api_key("aihubmix"),  # 换成你在 AiHubMix 生成的密钥
        base_url="https://api.aihubmix.com/v1",
    )

    # Using text-only input
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            },
        ],
        modalities=["text", "image"],
        temperature=0.7,
    )

    for part in response.choices[0].message.multi_mod_content:
        if "text" in part and part["text"] is not None:
            print(part["text"])
        elif "inline_data" in part and part["inline_data"] is not None:
            print("\n🖼️ [Image content received]")
            image_data = base64.b64decode(part["inline_data"]["data"])
            mime_type = part["inline_data"].get("mime_type", "image/png")
            print(f"Image type: {mime_type}")

            img = Image.open(BytesIO(image_data))
            img.save(save_path, "PNG")

            # 获取原始图像尺寸
            original_width, original_height = img.size

            # 计算新的尺寸（宽度和高度都缩小一半）
            new_width = original_width // 2
            new_height = original_height // 2

            # 使用 resize 方法改变图像尺寸
            # Image.BICUBIC 是一个常用的高质量缩放滤波器
            resized_img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)

            # 创建一个 BytesIO 对象来存储压缩后的图像数据
            compressed_image_buffer = BytesIO()

            # 将缩小尺寸后的图像保存到 BytesIO
            # 保存为 PNG 格式
            resized_img.save(compressed_image_buffer, format="PNG")

            # 将 BytesIO 的指针移到开始位置
            compressed_image_buffer.seek(0)

            # 将压缩（尺寸缩小）后的图像数据写入文件
            with open(f'{save_path[:-4]}_resize.png', "wb") as f:
                f.write(compressed_image_buffer.read())

            # 重新读取处理后的文件进行 base64 编码
            with open(f'{save_path[:-4]}_resize.png', "rb") as image_file:
                base64_encoder = base64.b64encode(image_file.read()).decode('utf-8')

            os.remove(f'{save_path[:-4]}_resize.png')

            return base64_encoder




def generate_image_gemini_edit(prompt,model,file_path,save_path):
    client = OpenAI(
        api_key=api_key("aihubmix"),  # 换成你在 AiHubMix 生成的密钥
        base_url="https://api.aihubmix.com/v1",
    )


    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    image = []
    for input_dir in file_path:
        image.append(input_dir)

    if not image:
        raise ValueError("No image files provided")

    max_size = 512  # 限制输入尺寸
    for img_path in image:
        img = Image.open(img_path)
        width, height = img.size
        # 检查并缩放图像
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)  # 使用 LANCZOS 滤波器进行缩放，效果较好
            img.save(img_path, "PNG")

    base64_image = encode_image(image[0])

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ],
        modalities=["text", "image"],
        temperature=0.7,
    )

    for part in response.choices[0].message.multi_mod_content:
        if "text" in part and part["text"] is not None:
            print(part["text"])

        # Process image content
        elif "inline_data" in part and part["inline_data"] is not None:
            image_data = base64.b64decode(part["inline_data"]["data"])

            img = Image.open(BytesIO(image_data))
            img.save(save_path, "PNG")

            # 获取原始图像尺寸
            original_width, original_height = img.size

            # 计算新的尺寸（宽度和高度都缩小一半）
            new_width = original_width // 2
            new_height = original_height // 2

            # 使用 resize 方法改变图像尺寸
            # Image.BICUBIC 是一个常用的高质量缩放滤波器
            resized_img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)

            # 创建一个 BytesIO 对象来存储压缩后的图像数据
            compressed_image_buffer = BytesIO()

            # 将缩小尺寸后的图像保存到 BytesIO
            # 保存为 PNG 格式
            resized_img.save(compressed_image_buffer, format="PNG")

            # 将 BytesIO 的指针移到开始位置
            compressed_image_buffer.seek(0)

            # 将压缩（尺寸缩小）后的图像数据写入文件
            with open(f'{save_path[:-4]}_resize.png', "wb") as f:
                f.write(compressed_image_buffer.read())

            # 重新读取处理后的文件进行 base64 编码
            with open(f'{save_path[:-4]}_resize.png', "rb") as image_file:
                base64_encoder = base64.b64encode(image_file.read()).decode('utf-8')

            os.remove(f'{save_path[:-4]}_resize.png')

            return base64_encoder



#base64_encoder = generate_image_gemini(prompt='头像 像素风 小猫咪震惊 头上三个！',model='gemini-2.0-flash-exp-image-generation',save_path='./1.png')
#v = generate_image_gemini_edit(prompt='修改为吉卜力风格',model='gemini-2.0-flash-exp-image-generation',file_path=['./8bd53d95df2d2ac1c30ec53a89226bd.jpg'],save_path='./sprite.png')
#a = generate_image_ideogram_remix(prompt='修改为戴眼镜的',model='weq',file_path=['2025-05-09-09-59-14.png'])
#
# pp=1

#print(generate_image_GPT(prompt='像素风 头像 小猫咪 赛博朋克 戴眼镜',model='gpt-4o-image-vip'))
    # url = "https://xiaohumini.site/v1/images/generations"
#generate_image_gpt_edits(prompt='修改为白色',model='gpt-image-1',file_path=['C:/Users/77684/Desktop/下载.png'])
#generate_image_GPT(prompt='像素风 头像 小猫咪 赛博朋克 戴眼镜',model='gpt-4o-image-vip')
# url = 'https://aihubmix.com/v1/images/generations'
# headers = {
#     'Content-Type': 'application/json',
#     'Authorization': api_key("aihubmix")
# }
#
# payload = json.dumps({
#     "model": 'gpt-image-1',
#     "prompt":'写实风格，女白领，高跟鞋，丝绸衬衫，在卫生间门前夹腿，要憋不住了',
#     "n": 1,
#     "size": "1024x1024",  # 生成图像的大小。必须是256x256、512x512或1024x1024for之一dall - e - 2。对于模型来说，必须是1024x1024、1792x1024
#     'quality': 'low',  # low high medium
#     'moderation': 'low',
#
# })
#
# response = requests.request("POST", url, headers=headers, data=payload)
# image_base64 = response.json()["data"][0]["b64_json"]
# image_bytes = base64.b64decode(image_base64)
#
# img = Image.open(BytesIO(image_bytes))
#
# # 获取原始图像尺寸
# original_width, original_height = img.size
#
# # 计算新的尺寸（宽度和高度都缩小一半）
# new_width = original_width // 2
# new_height = original_height // 2
#
# # 使用 resize 方法改变图像尺寸
# # Image.BICUBIC 是一个常用的高质量缩放滤波器
# resized_img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
#
# # 创建一个 BytesIO 对象来存储压缩后的图像数据
# compressed_image_buffer = BytesIO()
#
# # 将缩小尺寸后的图像保存到 BytesIO
# # 保存为 PNG 格式
# resized_img.save(compressed_image_buffer, format="PNG")
#
# # 将 BytesIO 的指针移到开始位置
# compressed_image_buffer.seek(0)
#
# # 将压缩（尺寸缩小）后的图像数据写入文件
# with open("1.png", "wb") as f:
#     f.write(compressed_image_buffer.read())
#
# # 重新读取处理后的文件进行 base64 编码
# with open("1.png", "rb") as image_file:
#     base64_encoder = base64.b64encode(image_file.read()).decode('utf-8')
#
# pp=1