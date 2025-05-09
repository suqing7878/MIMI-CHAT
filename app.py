from flask import Flask, request, jsonify, render_template, session, Response, send_from_directory,send_file
from flask_session import Session
import json
import os
from datetime import datetime
import base64
from datetime import datetime
from util import rename_filename
import uuid
from collections import defaultdict
from model import Chat_Microsoft,Chat_openai,Chat_DeepSeek,Chat_Qwen,Chat_Grok,Chat_Google,Chat_Claude,Chat_Other,Chat_DouBao,Chat_ZhiPu,Chat_Jina,Chat_ShangTang,Chat_MIMI,Chat_BaiDu,Chat_image_generate
from collections import defaultdict
from search import presearch_bool
# 全局存储流式响应数据
stream_responses = defaultdict(lambda: {"stream": None, "messages": None, "is_cancelled": False})


app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


history_folder = '/root/web2.0/mian_chat/history'

if not os.path.exists(history_folder):
    os.makedirs(history_folder)




@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def index(path):
    if path == "":
        return render_template('main.html')

    if os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return render_template('index.html')
    


# def index(path):   # 单机模式

#     if path != "" and os.path.exists(app.static_folder + '/' + path):
#         return send_from_directory(app.static_folder, path)
#     else:
#         return render_template('index.html')
    



@app.route('/start_chat', methods=['POST'])
def start_chat():
    user_message = request.json.get('message', '')  # 用户当前问题
    selected_model = request.json.get('model', '')  #模型名
    model_name = request.json.get('modelName', '')  # 厂家名
    file_path = request.json.get('filePath', '')  # 获取文件路径（list）
    history_file = request.json.get('historyFile', '')  # 当前用户的聊天历史文件名
    user_name = request.json.get('user_name', '')  # 用户名

    stream_id = str(uuid.uuid4())

    with open(f'{history_folder}/{user_name}/history/{history_file}.json', 'r', encoding='utf-8-sig') as file:
        messages = json.load(file)
   
    if len(messages)==0:
        filename = rename_filename(user_message)
        filename += datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        print('生成文件名：',filename)
    else:
        filename = history_file

    stream_responses[stream_id] = {
        "user_message": user_message,
        "messages": messages,
        "filename": filename,
        "user_name": user_name,
        'model_name': model_name,
        'selected_model': selected_model,
        'file_path':file_path,
        "is_cancelled": False  # 初始化取消标记
    }

    return jsonify({'stream_id': stream_id,"current_history": f"{filename}"})





@app.route('/chat/<stream_id>', methods=['GET'])
def chat(stream_id):
    def generate():
        stream_data = stream_responses.get(stream_id)

        filename = stream_data.get('filename', '')
        messages = stream_data.get('messages', '')
        user_message = stream_data.get('user_message', '')
        user_name = stream_data.get('user_name', '')
        model_name = stream_data.get('model_name', '')
        selected_model = stream_data.get('selected_model', '')
        file_path = stream_data.get('file_path', '')
        
        use_search = presearch_bool(user_message)
       
        if use_search:
            yield "data: ⚆_⚆ 疯狂Google中~\n\n"



        if model_name == '通义千问':
            stream, messages = Chat_Qwen(messages, user_message, selected_model, user_name, file_path,use_search)
        elif model_name == 'DeepSeek':
            stream, messages = Chat_DeepSeek(messages, user_message, selected_model, user_name, file_path,use_search)
        elif model_name == 'OpenAI':
            stream, messages = Chat_openai(messages, user_message, selected_model, user_name, file_path,use_search)
        elif model_name == 'Grok':
            stream, messages = Chat_Grok(messages, user_message, selected_model, user_name, file_path,use_search)
        elif model_name == 'Google':
            stream, messages = Chat_Google(messages, user_message, selected_model, user_name, file_path,use_search)
        elif model_name == 'Anthropic':
            stream, messages = Chat_Claude(messages, user_message, selected_model, user_name, file_path,use_search)
        elif model_name == 'Other':
            stream, messages = Chat_Other(messages, user_message, selected_model, user_name, file_path,use_search)
        elif model_name == '豆包':
            stream, messages = Chat_DouBao(messages, user_message, selected_model, user_name, file_path,use_search)
        elif model_name == '智谱':
            stream, messages = Chat_ZhiPu(messages, user_message, selected_model, user_name, file_path,use_search)
        elif model_name == 'Jina':
            stream, messages = Chat_Jina(messages, user_message, selected_model, user_name, file_path,use_search)
        elif model_name == '商汤':
            stream, messages = Chat_ShangTang(messages, user_message, selected_model, user_name, file_path,use_search)
        elif model_name == 'MIMI':
            stream, messages = Chat_MIMI(messages, user_message, selected_model, user_name, file_path,use_search)
        elif model_name == '百度':
            stream, messages = Chat_BaiDu(messages, user_message, selected_model, user_name, file_path,use_search)
        elif model_name == 'Microsoft':
            stream, messages = Chat_Microsoft(messages, user_message, selected_model, user_name, file_path,use_search)
        elif model_name =='图像生成':
            yield "data: ⚆_⚆ 超级努力的生成中~\n\n"
            base64_encoder, url,messages,image_save_path = Chat_image_generate(messages, user_message, selected_model, user_name, file_path, history_folder)

 
        history_file = f'{history_folder}/{user_name}/history/{filename}.json'  #先把输入存进去
        with open(history_file, mode='w', encoding='utf-8-sig') as f2:
            json.dump(messages, f2, ensure_ascii=False, indent=4)


        if model_name =='Jina':
            if stream is None:
                yield "data: Stream not found\n\n"
                return

            # 检查是否已被取消
            if stream_data["is_cancelled"]:
                yield "data: Stream cancelled\n\n"
                return
            
            bot_response = ""
            bot_think = ""

            think_bool = False
            for line in stream.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    # 处理SSE格式数据
                    if decoded_line.startswith('data:'):
                        event_data = decoded_line[5:].strip()  # 去掉"data:"前缀
                        if event_data != '[DONE]':
                            try:
                                chunk = json.loads(event_data)
                                # 类似OpenAI的流式输出结构
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    content = chunk['choices'][0].get('delta', {}).get('content', '')
                                    if content:
                                        if '<think>' in content:
                                            think_bool = True
                                        if '</think>' in content:
                                            think_bool = False
                                        if think_bool:
                
                                            bot_think += content
                                            print(f"Reasoning: {content}")
                                            reasoning_yield = content.replace('\n', '<br>')
                                            yield f"data: think:{reasoning_yield}\n\n"
                                            
                                        else:
                                            bot_response += content
                                            print('Content:', content)
                                            part_yield= content.replace('\n', '<br>')
                                            yield f"data: {part_yield}\n\n"
                            except json.JSONDecodeError:
                                pass

        elif model_name == '图像生成':

            bot_think = ""
            yield f"data: 无损图像下载链接：{url}\n\n"
            yield_file_path = image_save_path.split('/')[-1]
            yield f"data:image:data:image/png;base64,{base64_encoder};;{yield_file_path}\n\n"
            bot_response = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_encoder}"},
                    "file_path":image_save_path.split('/')[-1],
                },
                {"type": "text", "text": f"无损图像下载链接：{url}"},
            ]





        else:  # 正常模型
            if stream_data is None:
                yield "data: Stream not found\n\n"
                return


            bot_response = ""
            bot_think = ""

            for chunk in stream:
                # 检查是否已被取消
                if stream_data["is_cancelled"]:

                    if bot_think == "":
                        messages.append({"role": "assistant", "content": bot_response, 'modelName': model_name})
                    else:
                        messages.append({"role": "assistant", "content": bot_response, "think": bot_think, 'modelName': model_name})

                    history_file = f'{history_folder}/{user_name}/history/{filename}.json'
                    with open(history_file, mode='w', encoding='utf-8-sig') as f2:
                        json.dump(messages, f2, ensure_ascii=False, indent=4)
                        
                    yield "data: END\n\n"
                    return

                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    # 检查是否存在 reasoning_content 字段
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                        reasoning = delta.reasoning_content
                        bot_think += reasoning
                        print(f"{user_name},Reasoning: {reasoning}")
                        reasoning_yield = reasoning.replace('\n', '<br>')
                        yield f"data: think:{reasoning_yield}\n\n"
                    
                    if hasattr(delta, 'reasoning') and delta.reasoning is not None:
                        reasoning = delta.reasoning
                        bot_think += reasoning
                        print(f"{user_name},Reasoning: {reasoning}")
                        reasoning_yield = reasoning.replace('\n', '<br>')
                        yield f"data: think:{reasoning_yield}\n\n"

                    # 检查是否存在 content 字段
                    if hasattr(delta, 'content') and delta.content is not None:
                        part = delta.content
                        bot_response += part
                        print(f"{user_name},{selected_model},Content: {part}")
                        part_yield= part.replace('\n', '<br>')
                        if part_yield != '':
                            yield f"data: {part_yield}\n\n"
              
        if bot_think == "":
            messages.append({"role": "assistant", "content": bot_response, 'modelName': model_name})
        else:
            messages.append({"role": "assistant", "content": bot_response, "think": bot_think, 'modelName': model_name})

        history_file = f'{history_folder}/{user_name}/history/{filename}.json'
        with open(history_file, mode='w', encoding='utf-8-sig') as f2:
            json.dump(messages, f2, ensure_ascii=False, indent=4)

        yield "data: END\n\n"

        # 无论是否异常，流结束后都删除对应的 stream_id 条目
        if stream_id in stream_responses:
            del stream_responses[stream_id]
            print(f"Stream {stream_id} data cleaned up.")

    return Response(generate(), mimetype='text/event-stream')






#### 新建对话，需要返回新建对话的历史文件名（current_history），不带后缀和完整路径
@app.route('/create_conversation', methods=['POST'])
def create_conversation():
    user_name = request.json.get('user_name') # 用户名

    file_name = f"新对话"
    file_path = f'{history_folder}/{user_name}/history/{file_name}.json'
    new_history = []
    with open(file_path, 'w') as f:
        json.dump(new_history, f)
    
    return jsonify({"status": "success", "data": { "current_history": f"{file_name}"} }), 200





# ### 历史文件列表，需要返回该用户的所有历史文件名（history_list），不带后缀和完整路径
@app.route('/history_list', methods=['POST'])
def history_list():
   
    user_name = request.json.get('user_name')  # 用户名
    raw_history_list = os.listdir(f'{history_folder}/{user_name}/history/')
    
    # 存储文件名和创建时间的列表
    file_info_list = []

    for filename in raw_history_list:
        filepath = f'{history_folder}/{user_name}/history/{filename}'
        
        try:

            # 检查是否为 JSON 文件
            if filename.lower().endswith(".json") and filename !='新对话.json':
    
                # 获取文件的创建时间
                creation_time = os.path.getctime(filepath)
                file_info_list.append((filename, creation_time))
        except OSError as e:
            print(f"Error processing file {filename}: {e}")

    # 按创建时间从新到旧排序
    file_info_list.sort(key=lambda x: x[1], reverse=True)

    # 提取排序后的文件名（去掉后缀）
    sorted_history_list = [os.path.splitext(filename)[0] for filename, _ in file_info_list]

    return jsonify({"status": "success", "data": sorted_history_list}), 200





### 选择历史文件内容，需要返回该用户下的对应历史文件json形式的历史记录（history）
@app.route('/history', methods=['POST'])
def history():
    history_file = request.json.get('history_file') # 不带后缀的历史文件名
    user_name = request.json.get('user_name') # 用户名
    file_path = f'{history_folder}/{user_name}/history/{history_file}.json' 
    with open(file_path, 'r',encoding='utf-8-sig') as file:
        data = json.load(file)
        return jsonify({"status": "success", "data": { "history": data }}), 200
    





### 取消当前stream的回复
@app.route('/cancel-respond', methods=['POST'])
def cancel_respond():
    stream_id = request.json.get('stream_id')  # 当前 api stream generate 的 streamID

    stream_data = stream_responses.get(stream_id)
    if stream_data is None:
        return jsonify({"status": "error", "message": "Stream not found"}), 404

    # 设置取消标记
    stream_data["is_cancelled"] = True

    # 清理资源
    del stream_responses[stream_id]

    print("Generated has been canceled")
    return jsonify({"status": "success"}), 200





### 删除当前用户下的对应历史记录文件
@app.route('/delete_history', methods=['POST'])
def delete_history():
    delete_file_name = request.json.get('delete_file') # 不带后缀的file name
    user_name = request.json.get('user_name') # 用户名

    filepath = f'{history_folder}/{user_name}/history/{delete_file_name}.json'
    
    try:
        os.remove(filepath)
        print(f"File {filepath} has been deleted.")
    except OSError as e:
        print(f"Error deleting file {filepath}: {e}")

    return jsonify({"status": "success"}), 200





### 重命名当前用户下对应历史文件名，改成用户提供的新名字
@app.route('/rename_history', methods=['POST'])
def rename_history():
    rename_file = request.json.get('rename_file') # 不带后缀的file name (旧名字)
    new_name = request.json.get('new_name') # 不带后缀的file name （新名字）
    user_name = request.json.get('user_name') #用户名

    old_filepath = f'{history_folder}/{user_name}/history/{rename_file}.json'  
    new_filepath = f'{history_folder}/{user_name}/history/{new_name}.json'
    os.rename(old_filepath, new_filepath)

    return jsonify({"status": "success"}), 200





### 用户登陆，需要对应用户名列表
@app.route('/login', methods=['POST'])
def login():
    user_name = request.json.get('user_name') # 用户名

    if user_name not in ['admin']:
        return jsonify({"status": "false"}), 400
    
    # 处理login，不需要存名字
    user_history_folder = f'{history_folder}/{user_name}'
 
    if not os.path.exists(user_history_folder): # 登录时候创建用户目录
        os.makedirs(user_history_folder)
        os.makedirs(user_history_folder+'/history')
        os.makedirs(user_history_folder+'/file')
        
    return jsonify({"status": "success"}), 200





### 重新generate最后的ai问题，仅需要删除该历史记录中的最后一个ai生成的message和用户问题
@app.route('/regenerate', methods=['POST'])
def regenerate():
    user_name = request.json.get('user_name') # 用户名
    history_file = request.json.get('history_file') # 当前用户聊天的历史文件名称
    
    with open(f'{history_folder}/{user_name}/history/{history_file}.json', 'r', encoding='utf-8-sig') as file:
        data = json.load(file)

    if data[-1].get("role") == "assistant":
        # 查找并删除最后一个 "role": "assistant" 的对象
        for i in range(len(data) - 1, -1, -1):  # 从后往前遍历
            if data[i].get("role") == "assistant":
                del data[i]  # 删除符合条件的对象
                break  # 只删除最后一个符合条件的对象，然后退出循环
       
    # 将修改后的数据写回文件
    with open(f'{history_folder}/{user_name}/history/{history_file}.json', 'w', encoding='utf-8-sig') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    return jsonify({"status": "success"}), 200





@app.route('/upload_file', methods=['POST'])
def upload_file():
    user_name = request.form.get('user_name')
    file = request.files['file']

    if not request.files:
        print("No file upload")

    if 'file' not in request.files:
        print("No file part")
        return jsonify({"status": "error", "message": "No file part"}), 400
    
    if file.filename == '':
        print("No selected file")
        return jsonify({"status": "error", "message": "No selected file"}), 400
    
    if file:
        file_path = f'{history_folder}/{user_name}/file/{file.filename}' #
        file.save(file_path)
        return jsonify({"status": "success", "filePath": file_path}), 200




@app.route('/file-base64', methods=['POST'])
def get_file_base64():
    file_path = request.json.get('file_path') 
    
    with open(file_path, "rb") as f:
        file_data = f.read()
    
    base64_data = base64.b64encode(file_data).decode('utf-8')

    return jsonify({ "status": "success", "data": base64_data }), 200


@app.route('/reset_message', methods=['POST'])
def reset_message():
    current_conversation = request.json.get('current_conversation')  # 当前用户聊天的历史文件名称
    
    sender = request.json.get('sender')  # user / ai -- ai就是assistant
    reset_message= request.json.get('reset_message')
    user_name = request.json.get('user_name') # 用户名

    with open(f'{history_folder}/{user_name}/history/{current_conversation}.json', 'r', encoding='utf-8-sig') as file:
        data = json.load(file)
    
    
    if sender == 'ai':
        data[-1]['content'] = reset_message
    elif sender == 'user':
        if isinstance(data[-2]['content'], list):
            data[-2]['content'][-1] = {'text': reset_message, 'type': 'text'}
        else:
            data[-2]['content'] = reset_message
    with open(f'{history_folder}/{user_name}/history/{current_conversation}.json', 'w', encoding='utf-8-sig') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)  # indent 参数让输出更易读


    return jsonify({ "status": "success" }), 200


@app.route('/download', methods=['POST'])
def download_file():
    file_name = request.json.get('file_path').split('/')[-1]  #图片传回来的是file name   文档回来的是绝对路径
    user_name = request.json.get('user_name')
    print(file_name)
    file_path = f'{history_folder}/{user_name}/file/{file_name}'
    print(file_path )
    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": "File not found"}), 404

    print("下载")

    return send_file(
        file_path,
        as_attachment=True,
        download_name=os.path.basename(file_path)
    )

@app.route('/model_name', methods=['POST'])
def model_name():

    # 决定前端顺序
    modelButtons = [
                    {"key": "MIMI", "value": ['Mimi-Air',"Mimi-Pro"]},
                    {"key": "通义千问", "value": ["Qwen3","Qwen3-thinking",  "QwenVL-Max","QVQ" ]},
                    {"key": "OpenAI", "value": ['GPT-4.1-mini',"GPT-4.1", "o4-mini", "o3"]},
                    {"key": "Google", "value": ["Gemini-2.5-Pro", "Gemini-2.5-Flash-thinking"]},
                    {"key": "Grok", "value": ["Grok-3", "Grok-3-thinking"]},
                    {"key": "Microsoft", "value": ["Phi-4", "Phi-4-thinking"]},
                    {"key": "DeepSeek", "value": ["DeepSeek-V3", "DeepSeek-R1","DeepSeek-Prover"]},
                    {"key": "Anthropic", "value": ["Claude-3.7", "Claude-3.7-thinking"]},
                    {"key": "豆包", "value": ["Doubao-1.5",  "Doubao-1.5-thinking","Doubao-1.5-vision" ]},
                    {"key": "智谱", "value": ["GLM-4-Air",  "GLM-Z1-Air" ]},
                    {"key": "商汤", "value": ["SenseNova-Pro",  "SenseNova-thinking" ]},
                    {"key": "百度", "value": ["ERNIE-4.5",  "ERNIE-X1" ,  "ERNIE-4.5-VL"]},
                    {"key": "Jina", "value": ["Jina-deepsearch" ]},
                    {"key": "Other", "value": ["Llama-3.3-70B","Llama-4-Maverick","Mistral-Small-3.1-24B","Command-r-08", "Nova-Lite" ]},
                    {"key": "图像生成","value": ["GPT-image","GPT-image (图像编辑)","Gemini-image","Gemini-image (图像编辑)","Ideogram","Ideogram (图像编辑)", "Qwen","MiniMax", "DALL3", "Midjourney", "Flux"]},

  ]



    modelFileSupport = {
                        "MIMI": {
                            "Mimi-Air": ["all"],
                            "Mimi-Pro": ["all"]
                        },
                        "OpenAI": {
                            "GPT-4.1": ["all"],
                            "GPT-4.1-mini": ["all"],
                            "o4-mini": ["all"],
                            "o3": ["all"]
                        },
                        "DeepSeek": {
                            "DeepSeek-V3": ["document"],
                            "DeepSeek-R1": ["document"],
                            "DeepSeek-Prover": ["document"],
                        },
                        "Google": {
                            "Gemini-2.5-Pro": ["all"],
                            "Gemini-2.5-Flash-thinking": ["all"]
                        },
                        "Anthropic": {
                            "Claude-3.7": ["all"],
                            "Claude-3.7-thinking": ["all"]
                        },
                        "Microsoft": {
                            "Phi-4": ["all"],
                            "Phi-4-thinking": ["all"]
                        },
                        "Grok": {
                            "Grok-3": ["document"],
                            "Grok-3-thinking": ["document"]
                        },
                        "通义千问": {
                            "QWQ": ["document"],
                            "QVQ": ["all"],
                            "QwenVL-Max": ["all"],
                            "Qwen-Plus": ["document"],
                            "Qwen-Max": ["document"]
                        },
                        "豆包": {
                            "Doubao-1.5": ["document"],
                            "Doubao-1.5-thinking": ["document"],
                            "Doubao-1.5-vision": ["all"]
                        },
                        "智谱": {
                            "GLM-4-Air": ["document"],
                            "GLM-Z1-Air": ["document"]
                        },
                        "商汤": {
                            "SenseNova-Pro": ["all"],
                            "SenseNova-thinking": ["all"]
                        },
                        "百度": {
                            "ERNIE-4.5": ["document"],
                            "ERNIE-X1": ["document"],
                            "ERNIE-4.5-VL": ["all"]
                        },
                        "增强搜索": {
                            "Jina-deepsearch": ["all"]
                        },
                        "Kimi": {
                            "moonshot-v1-8k": ["document"],
                            "moonshot-v1-8k-vision-preview": ["image"]
                        },
                        "Other": {
                            "Mistral-Small-3.1-24B": ["document"],
                            "Command-r-08": ["document"],
                            "Nova-Lite": ["document"],
                            "Llama-3.3-70B": ["document"],
                            "Llama-4-Maverick": ["document"],
                            "Phi-4": ["document"]
                        },
                        "图像生成": {
                            "GPT-4o": ["document"],
                            "DALL3": ["document"],
                            "Midjourney": ["document"],

                            "Flux": ["document"],
                            "Qwen": ["document"],
                            "MiniMax": ["document"],
                            "GPT-image": ["document"],
                            "GPT-image (图像编辑)": ["image"],
                            "Ideogram": ["document"],
                            "Ideogram (图像编辑)": ["image"],
                            "Gemini-image": ["document"],
                            "Gemini-image (图像编辑)": ["image"],

                        }
                        }

    return jsonify({"status": "success", "model_name": modelButtons,"modelFileSupport":modelFileSupport}), 200

if __name__ == '__main__':
    #app.run(host='xxx.xxx.x.xx', port=xxxx, debug=False)
    app.run( debug=True)



