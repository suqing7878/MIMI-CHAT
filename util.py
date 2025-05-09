
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import os
import subprocess
import platform
from pathlib import Path
import base64
import requests
from api_key import api_key
from PIL import Image
import json
import openai
import os
import subprocess
import platform
from pathlib import Path
import copy
import io
def convert_to_pdf(input_file, output_dir=None):
    """

    sudo apt-get install libreoffice

    使用 LibreOffice 将 Word、Excel 或 PowerPoint 文件转换为 PDF
    支持中英文路径和文件名
    同时支持 Windows 和 Linux 系统

    参数:
        input_file: 输入文件路径
        output_dir: 输出目录，默认与输入文件相同

    返回:
        生成的 PDF 文件路径
    """
    input_file = os.path.abspath(input_file)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"找不到文件: {input_file}")

    # 获取输入文件信息
    file_path = Path(input_file)
    file_dir = file_path.parent
    file_name = file_path.stem
    file_ext = file_path.suffix.lower()

    # 检查文件类型
    supported_exts = [
        # Word 格式
        '.doc', '.docx', '.docm', '.dot', '.dotx', '.dotm', '.odt',
        # Excel 格式
        '.xls', '.xlsx', '.xlsm', '.xlt', '.xltx', '.xltm', '.ods','.csv',
        # PowerPoint 格式
        '.ppt', '.pptx', '.pptm', '.pot', '.potx', '.potm', '.odp'
    ]

    if file_ext not in supported_exts:
        raise ValueError(f"不支持的文件格式: {file_ext}。支持的格式有: {', '.join(supported_exts)}")
   
    output_dir_path = os.path.dirname(output_dir)
    # 生成的 PDF 文件名会和输入文件同名，后缀为 .pdf
    output_file_name = f"{file_name}.pdf"
    output_full_path = os.path.join(output_dir_path, output_file_name) # 生成 PDF 的完整输出路径

    # 根据操作系统确定 LibreOffice 可执行文件
    soffice_bin = 'soffice'
    if platform.system() == 'Windows':
        # Windows 上默认安装路径，可能需要根据实际情况调整
        program_files = os.environ.get('PROGRAMFILES', 'C:\\Program Files')
        libreoffice_paths = [
            os.path.join(program_files, 'LibreOffice', 'program', 'soffice.exe'),
            os.path.join(program_files, 'LibreOffice 7', 'program', 'soffice.exe'),
            os.path.join(program_files + ' (x86)', 'LibreOffice', 'program', 'soffice.exe'),
            os.path.join(program_files + ' (x86)', 'LibreOffice 7', 'program', 'soffice.exe')
        ]

        for path in libreoffice_paths:
            if os.path.exists(path):
                soffice_bin = f'"{path}"'
                break

    # 构建命令
    # 将 --outdir 参数设置为确定的输出目录 output_dir_path
    cmd = f'{soffice_bin} --headless --convert-to pdf --outdir "{output_dir_path}" "{input_file}"'

    try:
        process = subprocess.run(cmd, shell=True, check=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        return output_full_path # 返回生成的 PDF 文件的完整路径
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8', errors='ignore')
        print(f"转换失败，错误信息: {error_msg}")

        # 提供更详细的错误信息
        if "No such file or directory" in error_msg:
            print("LibreOffice 未找到。请确保已安装 LibreOffice 并在系统路径中。")
            if platform.system() == 'Windows':
                print("Windows 用户可以从 https://www.libreoffice.org/download/ 下载安装")
            elif platform.system() == 'Linux':
                print("Linux 用户可以使用包管理器安装，例如: sudo apt-get install libreoffice")

        raise


    
    
class TXTtoPDFConverter:
    # sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei ttf-wqy-zenhei fonts-noto-cjk
    def __init__(self):
        self.system = platform.system()
        self.font_registered = False
        self.register_fonts()

    def register_fonts(self):
        """根据操作系统注册中文字体"""
        font_paths = self.get_platform_font_paths()

        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font_name = os.path.splitext(os.path.basename(font_path))[0]
                    pdfmetrics.registerFont(TTFont(font_name, font_path))
                    self.font_name = font_name
                    self.font_registered = True
                    break
                except:
                    continue

        if not self.font_registered:
            try:
                # 最后尝试使用系统默认字体
                pdfmetrics.registerFont(TTFont("FallbackFont", "Helvetica"))
                self.font_name = "FallbackFont"
                print("警告: 使用备用字体，中文可能显示异常")
            except:
                raise Exception("无法注册任何字体，PDF生成失败")

    def get_platform_font_paths(self):
        """获取各平台常见中文字体路径"""
        common_fonts = []

        if self.system == "Windows":
            common_fonts.extend([
                "C:\\Windows\\Fonts\\simsun.ttc",  # 宋体
                "C:\\Windows\\Fonts\\msyh.ttc",  # 微软雅黑
                "C:\\Windows\\Fonts\\simhei.ttf",  # 黑体
            ])
        elif self.system == "Darwin":  # macOS
            common_fonts.extend([
                "/System/Library/Fonts/STHeiti Medium.ttc",  # 华文黑体
                "/System/Library/Fonts/Supplemental/Songti.ttc",  # 宋体
                "/Library/Fonts/Microsoft/SimSun.ttf",  # Windows字体移植
            ])
        else:  # Linux
            common_fonts.extend([
                "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",  # 文泉驿微米黑
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Noto字体
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # 文泉驿正黑
                "~/.fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc",  # 用户安装字体
            ])

        return common_fonts

    def convert(self, input_dir, output_dir):
        """
        执行转换

        :param input_txt_path: 输入TXT文件路径
        :param output_pdf_path: 输出PDF文件路径
        """
        # 创建PDF文档
        doc = SimpleDocTemplate(output_dir, pagesize=A4,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)

        # 定义样式
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='Normal_Center',
            parent=styles['Normal'],
            alignment=TA_CENTER,
            fontName=self.font_name,
            fontSize=12,
            leading=14
        ))
        styles.add(ParagraphStyle(
            name='Normal_Left',
            parent=styles['Normal'],
            alignment=TA_LEFT,
            fontName=self.font_name,
            fontSize=12,
            leading=14
        ))

        # 读取TXT文件内容（尝试多种编码）
        txt_content = self.read_txt_file( input_dir)

        # 准备PDF内容
        Story = []

        # 添加标题(使用文件名)
        title = os.path.splitext(os.path.basename( input_dir))[0]
        Story.append(Paragraph(title, styles['Normal_Center']))
        Story.append(Spacer(1, 12))

        # 添加正文
        paragraphs = txt_content.split('\n')
        for para in paragraphs:
            if para.strip():  # 跳过空行
                Story.append(Paragraph(para, styles['Normal_Left']))
                Story.append(Spacer(1, 12))

        # 生成PDF
        doc.build(Story)
        print(f"PDF文件已成功生成: {output_dir}")

    def read_txt_file(self, file_path):
        """尝试用多种编码读取TXT文件"""
        encodings = ['utf-8', 'gbk', 'gb18030', 'big5', 'utf-16']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        raise Exception(f"无法读取文件 {file_path}，尝试的编码: {', '.join(encodings)}")



def conver2pdf(input_dir,output_dir):
    '''
    
    sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei ttf-wqy-zenhei fonts-noto-cjk
    sudo apt-get install libreoffice
    '''
    file_type = None

    if input_dir.split('.')[-1].lower() in ['doc', 'docx', 'docm', 'dot', 'dotx', 'dotm', 'odt']:
        convert_to_pdf(input_dir, output_dir)
        pdf_content = read_pdf(output_dir)
        return pdf_content

    elif input_dir.split('.')[-1].lower() in ['xls','csv' ,'xlsx', 'xlsm', 'xlt', 'xltx', 'xltm', 'ods']:
        convert_to_pdf(input_dir, output_dir)
        pdf_content = read_pdf(output_dir)
        return pdf_content

    elif input_dir.split('.')[-1].lower() in ['ppt', 'pptx', 'pptm', 'pot', 'potx', 'potm', 'odp']:
        convert_to_pdf(input_dir, output_dir)
        pdf_content = read_pdf(output_dir)
        return pdf_content
    
    elif input_dir.split('.')[-1].lower() in ['txt']:
        with open(input_dir, 'r', encoding='utf-8-sig') as file:
            content = file.read()
        return content
    
    elif input_dir.split('.')[-1].lower() in ['py', 'mat', 'cpp', 'java', 'js', 'ts', 'html', 'css', 'php','sh', 'r', 'm', 'go', 'rs', 'swift','kt', 'dart', 'lua','html']:
        with open(input_dir, 'r', encoding='utf-8-sig') as file:
            content = file.read()
        return content
    
    elif input_dir.split('.')[-1].lower() in ['pdf']:
        file_type = 'pdf'

        pdf_content = read_pdf(input_dir)

        return pdf_content






def read_pdf(pdf_path):
    print('读取文件',pdf_path)
    def encode_file(image_path):
        with open(image_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8-sig")

    data = {
        "pdf": encode_file(f'{pdf_path}')
    }

    headers = {
        "Authorization": f"{api_key('jina')}",
        "Content-Type": "application/json"
    }

    response = requests.post("https://r.jina.ai/", json=data, headers=headers)

    if response.ok == True:
        return  response.text
    else:
        return ''
    




def convert_to_png(image_path, output_path=None, optimize=True, keep_transparency=True, max_size=(512, 512)):
    """
    将图片强制转换为PNG格式，并限制最大像素不超过指定大小。
    
    参数:
        image_path (str): 原始图片路径
        output_path (str): 输出文件路径(默认为原始目录，文件名改为 .png)
        optimize (bool): 是否优化PNG文件大小(默认True)
        keep_transparency (bool): 是否保留透明通道(默认True)
        max_size (tuple): 图片允许的最大宽度和高度，格式为 (width, height) (默认(512, 512))
    
    返回:
        str: 转换后的PNG文件路径
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"文件不存在: {image_path}")
    
    # 获取文件信息
    file_dir, file_name = os.path.split(image_path)
    file_base, file_ext = os.path.splitext(file_name)
    
    # 设置输出路径
    if output_path is None:
        output_path = os.path.join(file_dir, file_base + '.png')
    
    # 如果已经是png且不需要优化和缩放，直接返回原路径
    if file_ext.lower() == '.png' and not optimize and Image.open(image_path).size[0] <= max_size[0] and Image.open(image_path).size[1] <= max_size[1]:
        return image_path
    
    # 转换图片
    try:
        with Image.open(image_path) as img:
            # 处理缩放
            width, height = img.size
            if width > max_size[0] or height > max_size[1]:
                ratio = min(max_size[0] / width, max_size[1] / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS) # 使用高质量缩放算法
            
            # 处理透明通道
            if keep_transparency:
                if img.mode not in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGBA')
            else:
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                img = img.convert('RGB')
            
            # 保存为PNG
            img.save(output_path, 'PNG', optimize=optimize)
            
        return output_path
    except Exception as e:
        raise ValueError(f"图片转换失败: {str(e)}")




def rename_filename(messages):

    try:

        messages_input = [{
            "role": "user",
            "content":messages
           
        }]
        
        messages_input.append({
            "role": "user",
            "content":
            "请根据上文内容，为这段对话生成一个标题，要求：\n"
            "1. 标题长度≤10字符（含表情符号，如 😊、🎮 等）；\n"
            "2. 输出格式必须为 JSON，例如：{\"title\": \"😊 简短标题\"}；\n"
            "3. 禁止添加其他内容，仅返回 JSON 对象。\n"
            "4. 更加倾向使用中文。\n"
            "请确保键名使用英文双引号，值用双引号包裹。"
        })


        client = openai.OpenAI(
            api_key=api_key('qwen'),
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
        )

        respond = client.chat.completions.create(
                model='qwen-turbo',
                messages=messages_input,
                response_format={
                    'type': 'json_object'
                },
                stream=False
            )
        title = json.loads(respond.choices[0].message.content)['title']

        return title

    except:
        return '新对话'



# 使用示例
if __name__ == "__main__":

    # 使用示例
    png_path = convert_to_png(image_path = "1.jpg", output_path='C:/Users/yangy/Desktop/web/AI_Chat_pro_5001/2.png')
    print(f"转换后的PNG路径: {png_path}")


    # # 输入输出文件路径
    # input_txt = "服务器密码.txt"  # 替换为你的TXT文件路径
    # output_pdf = "output.pdf"  # 替换为你想要的输出路径

    # # 执行转换
    # converter = TXTtoPDFConverter()
    # converter.convert(input_txt, output_pdf)



    # # 转换 Word 文档
    # word_file = "2.doc"  # 支持中文文件名
    # convert_to_pdf(word_file)

    # # 转换 Excel 表格
    # excel_file = "1.xlsx"  # 支持中文文件名
    # convert_to_pdf(excel_file)

    # # 转换 PowerPoint 演示文稿
    # ppt_file = "看图工具教程(1).pptx"  # 支持中文文件名
    # convert_to_pdf(ppt_file,output_dir=None)

    # print(read_pdf('杨墨轩-符号搜索框架-申请书25-0311.pdf'))

