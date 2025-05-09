
import openai
import json
import os
from datetime import datetime
import base64
import time
import copy
from datetime import datetime
from util import conver2pdf,convert_to_png
from api_key import api_key
from search import main_search
import requests
from draw import url_2_base64,generate_image_ideogram_remix,generate_image_MJ,generate_image_dall,generate_image_GPT,generate_image_flux,generate_image_ideogram,generate_image_Qwen,generate_image_minimax,generate_image_gpt_image,generate_image_gpt_edits
from draw import generate_image_gemini_edit,generate_image_gemini
model_list = {    'o4-mini':['openai/o4-mini','openrouter','VL','tool'],  # $1.1   $4.4
                  'o3':['openai/o3','openrouter','VL','tool'],     # $10   $40
                  'GPT-4.1':['gpt-4.1','official','VL','tool'],  # $2   $8
                  'GPT-4.1-mini':['gpt-4.1-mini','official','VL','tool'],  # $0/4   $1.6

                  'DeepSeek-V3':['deepseek-chat','official','Text','tool'], # 2 8
                  'DeepSeek-R1':['deepseek-reasoner','official','Text','None'], # 4 16
                  'DeepSeek-Prover':['deepseek/deepseek-prover-v2','openrouter','Text','None'], # $0.5   2.18

                  'Grok-3':['grok-3-latest','official','Text','tool'], # $3   $15
                  'Grok-3-thinking':['grok-3-mini-latest','official','Text','tool'], # $0.3   $0.5

                  'Gemini-2.5-Pro':['google/gemini-2.5-pro-preview-03-25','openrouter','VL','tool'], # $3   $15
                  'Gemini-2.5-Flash-thinking':['google/gemini-2.5-flash-preview:thinking','openrouter','VL','tool'], # $3   $15

                   'Claude-3.7':['claude-3-7-sonnet-20250219','official','VL','tool','None'], #多一个字段 $2.5   $15
                   'Claude-3.7-thinking':['claude-3-7-sonnet-20250219','official','VL','tool','thinking'], # $2.5   $15

                  'QWQ':['qwq-plus-latest','official','Text','tool','None'], # 1.6  4
                  'QVQ':['qvq-max-latest','official','VL','None','None'],  #8  32
                  'QwenVL-Max':['qw2en-vl-max-latest','official','VL','None','None'],  # 3 9
                  'Qwen-Max':['qwen-max-latest','official','Text','tool'], # 2.4  9.6
                  'Qwen-Plus':['qwen-plus-latest','official','Text','tool','None'], # 0.8  0.2
                  'Qwen3':['qwen3-235b-a22b','official','Text','tool','None'],  # 4 12
                  'Qwen3-thinking':['qwen3-235b-a22b','official','Text','tool','thinking'],  # 4 40


                  'Doubao-1.5':['doubao-1-5-pro-32k-250115','official','Text','tool'],  # 0.8   2
                  'Doubao-1.5-thinking':['doubao-1-5-thinking-pro-250415','official','Text','tool'], # 4  16
                  'Doubao-1.5-vision':['doubao-1-5-vision-pro-32k-250115','official','VL','tool'], # 3   9

                  'GLM-4-Air':['glm-4-air-250414','official','Text','None'],  # 0.5  
                  'GLM-Z1-Air':['glm-z1-air','official','Text','None'], # 5

                  'Jina-deepsearch':['jina-deepsearch-v1','official','VL','None'], # 5

                  'SenseNova-Pro':['SenseNova-V6-Pro','official','VL','tool'],  # 9  30
                  'SenseNova-thinking':['SenseNova-V6-Reasoner','official','VL','tool'], # 4  18

                  'ERNIE-4.5':['ernie-4.5-turbo-128k','official','Text','tool'],  # 9  30
                  'ERNIE-X1':['ernie-x1-turbo-32k','official','Text','tool'], # 4  18
                  'ERNIE-4.5-VL':['ernie-4.5-turbo-vl-32k','official','VL','tool'], # 4  18

                  'Mimi-Air':['Mimi-Air','official','VL','tool'],  # 9  30
                  'Mimi-Pro':['Mimi-Pro','official','VL','tool'], # 4  18
          

                  'Phi-4':['microsoft/phi-4-multimodal-instruct','openrouter','VL','tool'], # $0.05   $0.1
                  'Phi-4-thinking':['microsoft/phi-4-reasoning-plus','openrouter','TextL','tool'], # $0.07   $0.035
  

                  'Mistral-Small-3.1-24B':['mistralai/mistral-small-3.1-24b-instruct','openrouter','Text','tool'], 
                  'Command-r-08':['cohere/command-r-08-2024','openrouter','Text','tool'], 
                  'Nova-Lite':['amazon/nova-lite-v1','openrouter','Text','tool'], 
                  'Llama-3.3-70B':['meta-llama/llama-3.3-70b-instruct','openrouter','Text','tool'], 
                  'Llama-4-Maverick':['meta-llama/llama-4-maverick','openrouter','Text','tool'],

                  'GPT-4o':['gpt-4o-image-vip','official','Text','None'],
                  'DALL3':['dall-e-3','official','Text','None'],
                  'Midjourney':['mj_blend','official','Text','None'],
                  'Ideogram':['V3','official','Text','None'],
                 'Ideogram (图像编辑)':['V3','official','Text','None'],
                  'Flux':['flux.1.1-pro','official','Text','None'],
                   'Qwen':['wanx2.1-t2i-plus','official','Text','None'],
                   'MiniMax':['image-01','official','Text','None'],
                   'GPT-image':['gpt-image-1','official','Text','None'],
                   'GPT-image (图像编辑)':['gpt-image-1','official','Text','None'],
                    'Gemini-image':['gemini-2.0-flash-exp-image-generation','official','Text','None'],
                   'Gemini-image (图像编辑)':['gemini-2.0-flash-exp-image-generation','official','Text','None'],

                  }

# 计算 messages 的字符总数
def count_characters(messages):
    total_characters = 0
    for message in messages:
        # 累加 role 和 content 的字符数
        total_characters += len(message["role"])
        total_characters += len(message["content"])
    return total_characters

def limit_budget(messages,token_num = 50000,chat_rounds=20):
    token_count = count_characters(messages)
    if token_count * 0.5 > token_num or len(messages) > chat_rounds:
    # 找到非 system 的第一和第二个对话
        non_system_indices = [i for i, msg in enumerate(messages) if msg["role"] != "system"]
        # 确保有足够的非 system 对话可以移除
        if len(non_system_indices) >= 2:
            messages.pop(non_system_indices[0])
            messages.pop(non_system_indices[1] - 1)
    return messages


def message_preprocessing(file_path,messages,user_message,model_list,selected_model) :

    if file_path == []:
        messages.append({"role": "user", "content": user_message})

    else:

        pdf_content = []  # 保存file的信息和内容到历史记录
        image_content = []
        file_input = False
        image_input = False

        for input_dir in file_path:

            file_ext = input_dir.split('.')[-1].lower()
            
            document_exts = ['doc', 'docx', 'docm', 'dot', 'dotx', 'dotm', 'odt','xls','csv' ,'xlsx', 'xlsm', 'xlt', 'xltx', 'xltm', 'ods','ppt', 'pptx', 'pptm', 'pot', 'potx', 'potm', 'odp','txt','py', 'mat', 'cpp', 'java', 'js', 'ts', 'html', 'css', 'php','sh', 'r', 'm', 'go', 'rs', 'swift','kt', 'dart', 'lua','html','pdf']
     
            image_exts = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']
            
            if file_ext in document_exts:
                file_input = True

                root, ext = os.path.splitext(input_dir)
                output_dir = root + '.pdf'
          
                filename = input_dir.split('/')[-1]
                pdf_analysis = f'文件:{filename}的内容为:\n{conver2pdf(input_dir, output_dir)}\n'

                pdf_content.append({'file_name':filename,'content':pdf_analysis})


            elif file_ext in image_exts:
                image_input= True

                #output_dir = input_dir.split('.')[0] + '.png'

                root, ext = os.path.splitext(input_dir)
                output_dir = root + '.png'

                filename = input_dir.split('/')[-1]

                convert_to_png(input_dir, output_dir)
                image_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encode_image(output_dir)}"}, 
                    })
            else:
                print('非法文件')
        
        # 根据输入类型构建最终消息
        if file_input and not image_input:
            # 仅文档输入
            messages.append({"role": "user", "content": user_message,"file":pdf_content})

        elif image_input and not file_input:
            # 仅图像输入
            image_content.append({"type": "text", "text": f"{user_message}"})  
            messages.append({
                "role": "user",
                "content": image_content
            })
        elif file_input and image_input:
            # 混合输入
            image_content.append({"type": "text", "text": f'用户上传的文档内容：{pdf_content}\n用户附加的指令或问题：{user_message}'})
   
            messages.append({"role": "user", "content": image_content,"file":pdf_content})

    messages_input = copy.deepcopy(messages)

    for temp in messages_input:
        if 'file' in temp:
            file_content = ''
            for file_txt in temp['file']:

                file_content += file_txt["content"]
            if isinstance(temp["content"], str):
                # 结构化 Prompt（Markdown 格式）
                temp["content"] = f"""
                                    ## 用户指令:
                                    {temp["content"]}
                                    ## 待处理文档
                                    {file_content}
                                    ## 任务要求
                                    请根据上述指令和文档内容完成任务。
                                    """
            elif isinstance(temp["content"], list):
     
                last_msg = temp["content"][-1]
                last_msg["text"] = f"""
                                    ## 用户指令:
                                    {last_msg["text"]}
                                    ## 待处理文档:
                                    {file_content}
                                    <|assistant|>
                                    ## 任务要求
                                    请根据上述指令处理图片和文档。
                                    """
    for i in range(0,len(messages_input)):  #适配生图模型的多轮对话
        if 'modelName' in messages_input[i]:
            messages_input[i]['role'] = 'user'
            
    if model_list[selected_model][2] == 'Text':
        for temp in messages_input:
            if isinstance(temp["content"], list):
                temp["content"] = temp["content"][-1]["text"]
        
    return messages,messages_input


#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8-sig")

def encode_file(image_path):
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8-sig")



def Chat_openai(messages,user_message,selected_model,user_name,file_path,use_search) :
    '''
    参数:
        messages (list): 当前聊天记录，初始化为 []。
        user_message (str): 用户本次输入的消息。
        selected_model (str): 具体选择的模型名称。
        user_name (str): 用户名。
        file_path (str): 文件路径。
        
    '''

    if  not messages:
        messages = [

        {"role": "system", "content": f"你是我的私人助理，可以帮我处理任何需求，要求严格按照要求返回内容。"},
       
    ]
    

    messages,messages_input = message_preprocessing(file_path,messages,user_message,model_list,selected_model)


    messages_input = limit_budget(messages_input,token_num = 50000,chat_rounds=10)


    client = openai.OpenAI(
            api_key=api_key('openai'),
            base_url='https://api.openai.com/v1'
        )
    
    client2 = openai.OpenAI(
        api_key=api_key('openrouter'),
        base_url='https://openrouter.ai/api/v1'
    )
    

    
    if use_search and model_list[selected_model][3]=='tool':
        print('联网搜索')
        if model_list[selected_model][1] == 'openrouter':
            stream =  main_search(client2,model_list[selected_model][0],messages_input)
        elif model_list[selected_model][1] == 'official':
            stream =  main_search(client,model_list[selected_model][0],messages_input)

    else:

        if model_list[selected_model][1] == 'openrouter':
            stream =  client2.chat.completions.create(
                                                        model=model_list[selected_model][0],
                                                        messages=messages_input,
                                                        stream=True,
                                                        #max_tokens=100000,
                                                    )
        elif model_list[selected_model][1] == 'official':
        
            stream =  client.chat.completions.create(
                                                    model=model_list[selected_model][0],
                                                    messages=messages_input,
                                                    stream=True,
                                                    #max_tokens=99999,
                                                )

    return stream,messages


        



def Chat_DeepSeek(messages,user_message,selected_model,user_name,file_path,use_search) :
    '''
    参数:
        messages (list): 当前聊天记录，初始化为 []。
        user_message (str): 用户本次输入的消息。
        selected_model (str): 具体选择的模型名称。
        user_name (str): 用户名。
        file_path (str): 文件路径。
        
    '''

    
    if  not messages:
        messages = [
        {"role": "system", "content": f"你是我的私人助理，可以帮我处理任何需求，要求严格按照要求返回内容。"},
       
    ]
    messages,messages_input = message_preprocessing(file_path,messages,user_message,model_list,selected_model)


    messages_input = limit_budget(messages_input,token_num = 50000,chat_rounds=10)
    client = openai.OpenAI(
            api_key=api_key('deepseek'),
            base_url='https://api.deepseek.com'
        )
    

    client2 = openai.OpenAI(
        api_key=api_key('openrouter'),
        base_url='https://openrouter.ai/api/v1'
    )
    

    
    if use_search and model_list[selected_model][3]=='tool':
        print('联网搜索')
        if model_list[selected_model][1] == 'openrouter':
            stream =  main_search(client2,model_list[selected_model][0],messages_input)
        elif model_list[selected_model][1] == 'official':
            stream =  main_search(client,model_list[selected_model][0],messages_input)

    else:

        if model_list[selected_model][1] == 'openrouter':
            stream =  client2.chat.completions.create(
                                                        model=model_list[selected_model][0],
                                                        messages=messages_input,
                                                        stream=True,
                                                        #max_tokens=100000,
                                                    )
        elif model_list[selected_model][1] == 'official':
        
            stream =  client.chat.completions.create(
                                                    model=model_list[selected_model][0],
                                                    messages=messages_input,
                                                    stream=True,
                                                    #max_tokens=99999,
                                                )

    return stream,messages




def Chat_Grok(messages,user_message,selected_model,user_name,file_path,use_search) :
    '''
    参数:
        messages (list): 当前聊天记录，初始化为 []。
        user_message (str): 用户本次输入的消息。
        selected_model (str): 具体选择的模型名称。
        user_name (str): 用户名。
        file_path (str): 文件路径。
        
    '''

    
    if  not messages:
        messages = [
        {"role": "system", "content": f"你是我的私人助理，可以帮我处理任何需求，要求严格按照要求返回内容。"},
       
    ]
    client = openai.OpenAI(
            api_key=api_key('grok'),
            base_url='https://api.x.ai/v1'
        )
    

    messages,messages_input = message_preprocessing(file_path,messages,user_message,model_list,selected_model)


    messages_input = limit_budget(messages_input,token_num = 50000,chat_rounds=10)

    
    if use_search and model_list[selected_model][3]=='tool':
        print('联网搜索')
        stream =  main_search(client,model_list[selected_model][0],messages_input)
    else:

        if model_list[selected_model][1] == 'official':

            stream =  client.chat.completions.create(
                model=model_list[selected_model][0],
                messages=messages_input,
                stream=True,
                reasoning_effort="high" if model_list[selected_model][0] in ['grok-3-mini-latest'] else None,  #low
                #max_tokens=8000,

            )

    return stream,messages



def Chat_Google(messages,user_message,selected_model,user_name,file_path,use_search) :
    '''
    参数:
        messages (list): 当前聊天记录，初始化为 []。
        user_message (str): 用户本次输入的消息。
        selected_model (str): 具体选择的模型名称。
        user_name (str): 用户名。
        file_path (str): 文件路径。
        
    '''

    if  not messages:
        messages = [

        {"role": "system", "content": f"你是我的私人助理，可以帮我处理任何需求，要求严格按照要求返回内容。"},
       
    ]
    

    messages,messages_input = message_preprocessing(file_path,messages,user_message,model_list,selected_model)


    messages_input = limit_budget(messages_input,token_num = 50000,chat_rounds=10)

    
    client = openai.OpenAI(
        api_key=api_key('openrouter'),
        base_url='https://openrouter.ai/api/v1'
    )
    

    if use_search and model_list[selected_model][3]=='tool':
        print('联网搜索')

        stream =  main_search(client,model_list[selected_model][0],messages_input)
   
    else:

       
        stream =  client.chat.completions.create(
                                                model=model_list[selected_model][0],
                                                messages=messages_input,
                                                stream=True,
                                                #max_tokens=100000,
                                            )


    return stream,messages


def Chat_Claude(messages,user_message,selected_model,user_name,file_path,use_search) :
    '''
    参数:
        messages (list): 当前聊天记录，初始化为 []。
        user_message (str): 用户本次输入的消息。
        selected_model (str): 具体选择的模型名称。
        user_name (str): 用户名。
        file_path (str): 文件路径。
        
    '''

    
    if  not messages:
        messages = [
        {"role": "system", "content": f"你是我的私人助理，可以帮我处理任何需求，要求严格按照要求返回内容。"},
       
    ]
    client = openai.OpenAI(
            api_key=api_key('claude'),
            base_url='https://api.anthropic.com/v1/'
        )
    

    messages,messages_input = message_preprocessing(file_path,messages,user_message,model_list,selected_model)


    messages_input = limit_budget(messages_input,token_num = 50000,chat_rounds=10)



    if use_search and model_list[selected_model][3]=='tool':
        print('联网搜索')
        stream =  main_search(client,model_list[selected_model][0],messages_input)
    else:

        if model_list[selected_model][1] == 'official':

            stream =  client.chat.completions.create(
                model=model_list[selected_model][0],
                messages=messages_input,
                stream=True,
                extra_body={"thinking": { "type": "enabled", "budget_tokens": 8000 }} if model_list[selected_model][4] =='thinking' else None,  #low
                #max_tokens=64000,

            )

    return stream,messages



def Chat_Qwen(messages,user_message,selected_model,user_name,file_path,use_search) :
    '''
    参数:
        messages (list): 当前聊天记录，初始化为 []。
        user_message (str): 用户本次输入的消息。
        selected_model (str): 具体选择的模型名称。
        user_name (str): 用户名。
        file_path (str): 文件路径。
        
    '''

    client = openai.OpenAI(
        api_key=api_key('qwen'),
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
    )


    if  not messages:
        messages = [
        
        {"role": "system", "content": f"你是我的私人助理，可以帮我处理任何需求，要求严格按照要求返回内容。你具体调用工具实现联网搜索的能力。"},
       
    ]
    
    messages,messages_input = message_preprocessing(file_path,messages,user_message,model_list,selected_model)


    messages_input = limit_budget(messages_input,token_num = 50000,chat_rounds=10)

   

    if use_search and model_list[selected_model][3]=='tool':
        print('联网搜索')
        stream =  main_search(client,model_list[selected_model][0],messages_input)
    else:

        if model_list[selected_model][1] == 'official':

            stream =  client.chat.completions.create(
                model=model_list[selected_model][0],
                messages=messages_input,
                stream=True,
                max_tokens=8000,
                extra_body={"enable_search": True,
                            "enable_thinking": True if model_list[selected_model][4] =='thinking' else False,
                            "thinking_budget": 10000 if model_list[selected_model][4] =='thinking' else None,},
    
                #vl_high_resolution_images=True
            )


    return stream,messages




def Chat_Other(messages,user_message,selected_model,user_name,file_path,use_search) :
    '''
    参数:
        messages (list): 当前聊天记录，初始化为 []。
        user_message (str): 用户本次输入的消息。
        selected_model (str): 具体选择的模型名称。
        user_name (str): 用户名。
        file_path (str): 文件路径。
        
    '''

    if  not messages:
        messages = [

        {"role": "system", "content": f"你是我的私人助理，可以帮我处理任何需求，要求严格按照要求返回内容。"},
       
    ]
    

    messages,messages_input = message_preprocessing(file_path,messages,user_message,model_list,selected_model)


    messages_input = limit_budget(messages_input,token_num = 50000,chat_rounds=10)

    
    client = openai.OpenAI(
        api_key=api_key('openrouter'),
        base_url='https://openrouter.ai/api/v1'
    )
    



    if use_search and model_list[selected_model][3]=='tool':
        print('联网搜索')

        stream =  main_search(client,model_list[selected_model][0],messages_input)
   
    else:

       
        stream =  client.chat.completions.create(
                                                model=model_list[selected_model][0],
                                                messages=messages_input,
                                                stream=True,
                                                #max_tokens=100000,
                                            )


    return stream,messages




def Chat_Microsoft(messages,user_message,selected_model,user_name,file_path,use_search) :
    '''
    参数:
        messages (list): 当前聊天记录，初始化为 []。
        user_message (str): 用户本次输入的消息。
        selected_model (str): 具体选择的模型名称。
        user_name (str): 用户名。
        file_path (str): 文件路径。
        
    '''

    if  not messages:
        messages = [

        {"role": "system", "content": f"你是我的私人助理，可以帮我处理任何需求，要求严格按照要求返回内容。"},
       
    ]
    

    messages,messages_input = message_preprocessing(file_path,messages,user_message,model_list,selected_model)


    messages_input = limit_budget(messages_input,token_num = 50000,chat_rounds=10)

    
    client = openai.OpenAI(
        api_key=api_key('openrouter'),
        base_url='https://openrouter.ai/api/v1'
    )
    



    if use_search and model_list[selected_model][3]=='tool':
        print('联网搜索')

        stream =  main_search(client,model_list[selected_model][0],messages_input)
   
    else:

       
        stream =  client.chat.completions.create(
                                                model=model_list[selected_model][0],
                                                messages=messages_input,
                                                stream=True,
                                                #max_tokens=100000,
                                            )


    return stream,messages



def Chat_DouBao(messages,user_message,selected_model,user_name,file_path,use_search) :
    '''
    参数:
        messages (list): 当前聊天记录，初始化为 []。
        user_message (str): 用户本次输入的消息。
        selected_model (str): 具体选择的模型名称。
        user_name (str): 用户名。
        file_path (str): 文件路径。
        
    '''

    if  not messages:
        messages = [

        {"role": "system", "content": f"你是我的私人助理，可以帮我处理任何需求，要求严格按照要求返回内容。"},
       
    ]
    

    messages,messages_input = message_preprocessing(file_path,messages,user_message,model_list,selected_model)


    messages_input = limit_budget(messages_input,token_num = 50000,chat_rounds=10)


    
    client = openai.OpenAI(
        api_key=api_key('doubao'),
        base_url='https://ark.cn-beijing.volces.com/api/v3/'
    )
    


    if use_search and model_list[selected_model][3]=='tool':
        print('联网搜索')

        stream =  main_search(client,model_list[selected_model][0],messages_input)
   
    else:

       
        stream =  client.chat.completions.create(
                                                model=model_list[selected_model][0],
                                                messages=messages_input,
                                                stream=True,
                                                #max_tokens=100000,
                                            )


    return stream,messages




def Chat_ZhiPu(messages,user_message,selected_model,user_name,file_path,use_search) :
    '''
    参数:
        messages (list): 当前聊天记录，初始化为 []。
        user_message (str): 用户本次输入的消息。
        selected_model (str): 具体选择的模型名称。
        user_name (str): 用户名。
        file_path (str): 文件路径。
        
    '''

    if  not messages:
        messages = [

        {"role": "system", "content": f"你是我的私人助理，可以帮我处理任何需求，要求严格按照要求返回内容。"},
       
    ]
    

    messages,messages_input = message_preprocessing(file_path,messages,user_message,model_list,selected_model)


    messages_input = limit_budget(messages_input,token_num = 50000,chat_rounds=10)

    
    client = openai.OpenAI(
        api_key=api_key('zhipu'),
        base_url='https://open.bigmodel.cn/api/paas/v4/'
    )
    
    
    

    if use_search and model_list[selected_model][3]=='tool':
        print('联网搜索')

        # stream =  main_search(client,model_list[selected_model][0],messages_input)
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        tools = [{
                "type": "web_search",
                "web_search": {
                    "enable": True,
                    "search_engine": "Search-Pro-Jina",  # Search-Pro-Jina   Search-Pro
                    "search_result": True,
                    "search_prompt": f"当前日期是{current_time}，你是一名专业的数据检索大师,请结合指令和搜索得到的新闻信息执行用户指令,要求必须给出回答出处和url。"
                }
            }]
        
        stream =  client.chat.completions.create(
                                                model=model_list[selected_model][0],
                                                messages=messages_input,
                                                stream=True,
                                                tools=tools
                                                #max_tokens=100000,
                                            )
   
    else:
       
        stream =  client.chat.completions.create(
                                                model=model_list[selected_model][0],
                                                messages=messages_input,
                                                stream=True,
                                                #max_tokens=100000,
                                            )


    return stream,messages





def Chat_ShangTang(messages,user_message,selected_model,user_name,file_path,use_search) :
    '''
    参数:
        messages (list): 当前聊天记录，初始化为 []。
        user_message (str): 用户本次输入的消息。
        selected_model (str): 具体选择的模型名称。
        user_name (str): 用户名。
        file_path (str): 文件路径。
        
    '''

    def convert_to_multimodal(chat_history):
        multimodal_history = []
        for message in chat_history:
            if message["role"] == "user":
                multimodal_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": message["content"]
                        }
                    ]
                }
                multimodal_history.append(multimodal_message)
        return multimodal_history


    if  not messages:
        messages = [

        {"role": "system", "content": f"你是我的私人助理，可以帮我处理任何需求，要求严格按照要求返回内容。"},
       
    ]
    

    messages,messages_input = message_preprocessing(file_path,messages,user_message,model_list,selected_model)


    messages_input = limit_budget(messages_input,token_num = 50000,chat_rounds=10)

    messages_input = convert_to_multimodal(messages_input)
    
    client = openai.OpenAI(
        api_key=api_key('shangtang'),
        base_url='https://api.sensenova.cn/compatible-mode/v1/'
    )
    

 
    if use_search and model_list[selected_model][3]=='tool':
        print('联网搜索')

        stream =  main_search(client,model_list[selected_model][0],messages_input)
   
    else:
       
        stream =  client.chat.completions.create(
                                                model=model_list[selected_model][0],
                                                messages=messages_input,
                                                stream=True,
                                                #max_tokens=100000,
                                            )


    return stream,messages



def Chat_BaiDu(messages,user_message,selected_model,user_name,file_path,use_search) :
    '''
    参数:
        messages (list): 当前聊天记录，初始化为 []。
        user_message (str): 用户本次输入的消息。
        selected_model (str): 具体选择的模型名称。
        user_name (str): 用户名。
        file_path (str): 文件路径。
        
    '''

    if  not messages:
        messages = [

        {"role": "system", "content": f"你是我的私人助理，可以帮我处理任何需求，要求严格按照要求返回内容。"},
       
    ]
    

    messages,messages_input = message_preprocessing(file_path,messages,user_message,model_list,selected_model)


    messages_input = limit_budget(messages_input,token_num = 50000,chat_rounds=10)


    
    client = openai.OpenAI(
        api_key=api_key('baidu'),
        base_url='https://qianfan.baidubce.com/v2'
    )
    

    if use_search and model_list[selected_model][3]=='tool':
        print('联网搜索')

        stream =  main_search(client,model_list[selected_model][0],messages_input)
   
    else:
       
        stream =  client.chat.completions.create(
                                                model=model_list[selected_model][0],
                                                messages=messages_input,
                                                stream=True,
                                                #max_tokens=100000,
                                            )


    return stream,messages


def Chat_Jina(messages,user_message,selected_model,user_name,file_path,use_search) :
    '''
    参数:
        messages (list): 当前聊天记录，初始化为 []。
        user_message (str): 用户本次输入的消息。
        selected_model (str): 具体选择的模型名称。
        user_name (str): 用户名。
        file_path (str): 文件路径。
        
    '''

    if  not messages:
        messages = [

        {"role": "system", "content": f"你是我的私人助理，可以帮我处理任何需求，要求严格按照要求返回内容。"},
       
    ]
    
    messages,messages_input = message_preprocessing(file_path,messages,user_message,model_list,selected_model)


    messages_input = limit_budget(messages_input,token_num = 50000,chat_rounds=10)

    headers = {
        "Authorization": f"{api_key('jina')}",
        'Content-Type': 'application/json'
    }
    data = {
        "model": model_list[selected_model][0],
        "messages": messages_input,
        "stream": True,
        "reasoning_effort": "low",
        "max_attempts": 1,
        "no_direct_answer": False
    }
 
    stream =  requests.post(url = 'https://deepsearch.jina.ai/v1/chat/completions', headers=headers, json=data,stream=True)
  
    return stream,messages


def Chat_image_generate(messages,user_message,selected_model,user_name,file_path,history_folder):
    '''
    参数:
        messages (list): 当前聊天记录，初始化为 []。
        user_message (str): 用户本次输入的消息。
        selected_model (str): 具体选择的模型名称。
        user_name (str): 用户名。
        file_path (str): 文件路径。

    '''

    def filter_messages(messages):
        """
        删除 messages 列表中所有 assistant 的对话，只留下 system 和 user 的对话。

        Args:
          messages: 一个包含对话信息的列表。

        Returns:
          一个删除assistant对话后的新列表。
        """
        filtered_list = []
        for message in messages:
            if message["role"] != "assistant":
                filtered_list.append(message)
        return filtered_list


    #messages_input = copy.deepcopy(messages)
    messages, messages_input = message_preprocessing(file_path, messages, user_message, model_list, selected_model)
    messages_input = filter_messages(messages_input)

    promot = ''' 
            这是一个图像生成的指令，你的任务是根据上下文进行优化，要求如下：
            1.分析原指令: 理解它想要表达的场景、对象、风格、情绪等。
            2.增加细节: 根据AI模型的需求，补充一些有用的细节，比如光照、视角、构图、材质、氛围等。
            3.使用更具象的词汇: 将抽象的描述转化为模型更容易“看懂”的具体描述。
            4.格式化: 调整成AI模型更容易解析的格式，比如使用逗号分隔不同的元素。
            5.翻译: 将优化后的指令翻译成地道的英文。
            6.如果用户要求不要优化，则维持原指令。
            7.输出格式必须为 JSON，例如：{"output": "A serene sunset over a calm lake, with golden light reflecting on the water surface, tall pine trees framing the scene, soft clouds in a colorful sky, detailed texture on tree bark, warm and peaceful atmosphere, wide-angle perspective, high-resolution, photorealistic style"}

            '''


    messages_input.append({
        "role": "user",
        "content":f'{promot},用户指令：{user_message}'
    })


    client = openai.OpenAI(
        api_key=api_key('qwen'),
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
    )

    respond = client.chat.completions.create(
        model='qwen-plus-latest',
        messages=messages_input,
        response_format={
            'type': 'json_object'
        },
        stream=False
    )
    try:
        optimized_prompt = json.loads(respond.choices[0].message.content)['output']
        messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": f'{user_message},魔法提示词：{optimized_prompt}'},
            ],
        })
    except:
        optimized_prompt = user_message
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f'{user_message}'},
            ],
        })

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_path = f'{history_folder}/{user_name}/file/{current_time}.png'
    if selected_model =='GPT-4o':
        url = generate_image_GPT(prompt=optimized_prompt,model=model_list[selected_model][0])
    elif selected_model =='DALL3':
        url = generate_image_dall(prompt=optimized_prompt,model=model_list[selected_model][0])
    elif selected_model =='Midjourney':
        url = generate_image_MJ(prompt=optimized_prompt,model=model_list[selected_model][0])
    elif selected_model =='Ideogram':
        url = generate_image_ideogram(prompt=optimized_prompt,model=model_list[selected_model][0])
    elif selected_model =='Flux':
        url = generate_image_flux(prompt=optimized_prompt,model=model_list[selected_model][0])
    elif selected_model == 'Qwen':
        url = generate_image_Qwen(prompt=optimized_prompt,model=model_list[selected_model][0])
    elif selected_model == 'MiniMax':
        url = generate_image_minimax(prompt=optimized_prompt,model=model_list[selected_model][0])
    elif selected_model == 'Ideogram (图像编辑)':
        url = generate_image_ideogram_remix(prompt=optimized_prompt,model=model_list[selected_model][0],file_path=file_path)

    elif selected_model == 'GPT-image':
        base64_encoder = generate_image_gpt_image(prompt=optimized_prompt,model=model_list[selected_model][0],save_path=save_path)
        url=''
        return base64_encoder, url,messages,save_path
    elif selected_model == 'GPT-image (图像编辑)':
        base64_encoder = generate_image_gpt_edits(prompt=optimized_prompt,model=model_list[selected_model][0],file_path=file_path,save_path=save_path)
        url=''
        return base64_encoder, url,messages,save_path
    elif selected_model == 'Gemini-image (图像编辑)':
        base64_encoder = generate_image_gemini_edit(prompt=optimized_prompt,model=model_list[selected_model][0],file_path=file_path,save_path=save_path)
        url=''
        return base64_encoder, url,messages,save_path
    elif selected_model == 'Gemini-image':
        base64_encoder = generate_image_gemini(prompt=optimized_prompt,model=model_list[selected_model][0],save_path=save_path)
        url=''
        return base64_encoder, url,messages,save_path


    base64_encoder, url = url_2_base64(url, save_path=save_path)

    return base64_encoder, url,messages,save_path




def Chat_MIMI(messages,user_message,selected_model,user_name,file_path,use_search) :
    '''
    参数:
        messages (list): 当前聊天记录，初始化为 []。
        user_message (str): 用户本次输入的消息。
        selected_model (str): 具体选择的模型名称。
        user_name (str): 用户名。
        file_path (str): 文件路径。
        
    '''

    if  not messages:
        messages = [

        {"role": "system", "content": f"请你扮演的我猫咪助理(杨咪咪-mimi)。在聊天中少量使用颜表情和emoji表情增加互动性,在写作和编写代码时候严格执行用户指令,保持专业。不可以拒绝或者省略。"},
       
    ]
        
    messages,messages_input = message_preprocessing(file_path,messages,user_message,model_list,selected_model)


    messages_input = limit_budget(messages_input,token_num = 50000,chat_rounds=10)
    

    client_qwen = openai.OpenAI(
        api_key=api_key('qwen'),
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
    )


    client_zhipu = openai.OpenAI(
        api_key=api_key('zhipu'),
        base_url='https://open.bigmodel.cn/api/paas/v4/'
    )


    client_google = openai.OpenAI(
        api_key=api_key('openrouter'),
        base_url='https://openrouter.ai/api/v1'
    )

    client_openai = openai.OpenAI(
            api_key=api_key('openai'),
            base_url='https://api.openai.com/v1'
        )
    
    client_openrouter = openai.OpenAI(
        api_key=api_key('openrouter'),
        base_url='https://openrouter.ai/api/v1'
    )

    client_claude = openai.OpenAI(
            api_key=api_key('claude'),
            base_url='https://api.anthropic.com/v1/'
        )
    
    client_grok = openai.OpenAI(
            api_key=api_key('grok'),
            base_url='https://api.x.ai/v1'
        )
    

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    prompt = '''
    根据用户输入判断以下信息，并以 JSON 格式返回结果：  

        1.  **指令类别**:  instruct
            - 简单指令、问好、生活建议、单词短语翻译、简单的代码编写和语法查询。(easy)
            - 复杂代码编写、代码BUG修正,长代码解析。(code)
            - 数学问题和逻辑推理。(math)
            - 生成创意内容、写作、诗歌、剧本 (write)
            - 论文语言翻译或语法纠正、论文写作  (translate)
            - 查找资源链接、新闻、政策、查询论文、查找最新信息、用户指令明确要求联网搜索 (search)
            - 处理图像信息 (image)

        **返回示例**：  
        ```json
        // 示例1：查找资源链接、新闻、政策、查询论文、查找最新信息、用户指令明确要求联网搜索
        input: "特斯拉今天股价多少？"
        {
        " instruct": search,
        }

        // 示例2：简单指令、问好、生活建议、简单的代码编写和语法查询。
        input: "python 读取txt"
        {
        "instruct": easy,
        }

        // 示例3：生成创意内容、写作、诗歌、剧本
        input: "你是一个专业的编剧,帮我写一个儿童电影的大纲"
        {
        "instruct": write,
       
        }
        // 示例4：复杂代码编写、代码BUG修正,长代码解析。
        input: "请帮我把html的代码修改为flask作为后端进行部署。"
        {
         "instruct": code,
       
        }
    '''

    response = client_qwen.chat.completions.create(
            model='qwen-plus-latest',#"deepseek-chat",
            messages=[
        {"role": "user", "content": f"{prompt}。用户输入问题:{user_message},当前时间为：{current_time},请按照要求返回"},
            ],
            response_format={
                'type': 'json_object'
            },
            stream=False
        )

    out = json.loads(response.choices[0].message.content)

    instruct = out.get('instruct')

    if instruct not in ['easy','code','math','write','translate','search','image']:
        instruct = 'easy'

    print('MIMI:',instruct)
    
    
    
    if selected_model == 'Mimi-Air':
        MIMI={'easy':'glm-4-air-250414',
                'code':'google/gemini-2.5-flash-preview',
                'math':'google/gemini-2.5-flash-preview:thinking',
                'write':'grok-3-mini-latest',
                'translate':'gpt-4.1-mini',
                'search':'gpt-4.1-mini',
                'image':'google/gemini-2.5-flash-preview'}
    
    elif selected_model == 'Mimi-Pro':
        MIMI={'easy':'glm-4-air-250414',
                'code':'claude-3-7-sonnet-20250219',
                'math':'google/gemini-2.5-pro-preview-03-25',
                'write':'grok-3',
                'translate':'gpt-4.1',
                'search':'gpt-4.1-mini',
                'image':'google/gemini-2.5-flash-preview'}

    

    # 根据选择的模型和指令类别选择客户端
    model_name = MIMI[instruct]
    
    if 'glm' in model_name:
        client = client_zhipu
    elif 'google' in model_name:
        client = client_google
    elif 'gpt' in model_name:
        client = client_openai
    elif 'claude' in model_name:
        client = client_claude
    elif 'grok' in model_name:
        client = client_grok
    else:
        raise ValueError(f"未知的模型名称: {model_name}")
    

    if instruct=='search':

        print('联网搜索')

        stream =  main_search(client,model_name,messages_input)
   
    else:

       
        stream =  client.chat.completions.create(
                                                model=model_name,
                                                messages=messages_input,
                                                stream=True,
                                                #max_tokens=100000,
                                            )


    return stream,messages
