
from datetime import datetime
import json
import openai
from api_key import api_key
import requests
import json
from typing import Dict, Any
from exa_py import Exa
import requests

#client = openai.OpenAI(api_key=f"{api_key('deepseek')}", base_url="https://api.deepseek.com")
client = openai.OpenAI(api_key=f"{api_key('qwen')}", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

exa = Exa(api_key = f"{api_key('exa')}")
bocha_headers = {
    'Authorization': f"{api_key('bocha')}",
    'Content-Type': 'application/json'
    }

jina_headers = {
        "Accept": "application/json",
        "Authorization": f"{api_key('jina')}",
        "X-Respond-With": "no-content"
    }

def presearch_bool(user_message):

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    prompt = '''
    根据用户输入判断以下信息，并以 JSON 格式返回结果：  

        1. **是否需要调用搜索引擎**（`use_search`，`true`/`false`）  
        - **需要调用搜索引擎的情况**：  
            - 查找最新信息（科技动态、体育赛事结果等）  
            - 获取实时数据（股票价格、天气预报等）  
            - 研究特定主题（学术论文、行业报告等）  
            - 验证事实或数据（历史事件日期、科学定理等）  
            - 获取用户评价（产品评测、餐厅评分等）  
            - 查找资源链接（软件下载、特定网站等）  
            - 查询最新政策（法律法规、政府公告等） 
            - 用户指令明确要求（联网搜索，帮我查一下等）
        - **不需要调用搜索引擎的情况**：  
            - 基于已有知识回答问题（概念解释、术语定义等）  
            - 数学计算或逻辑推理（解题、谜题等）  
            - 生成创意内容（写作、诗歌、文案等）  
            - 提供个人建议（学习方法、生活建议等）  
            - 分析/总结已有文件（文档解析、摘要等）  
            - 语言翻译或语法纠正  
            - 代码编写或算法执行  (严格遵守，涉及到代码不要搜索)
            - 代码报错问题，代码语法  (严格遵守，涉及到代码不要搜索)

        **返回示例**：  
        ```json
        // 示例1：实时信息（高时效性，少量结果）  
        input: "特斯拉今天股价多少？"
        {
        "use_search": true,
        }

        // 示例2：学术研究（广泛覆盖，国际引擎）  
        input: "推荐10篇关于大语言模型的论文"
        {
        "use_search": true,
        }

        // 示例3：无需搜索（已有知识）  
        input: "Python的lambda函数怎么写？"
        {
        "use_search": false,
       
        }
        // 示例4：用户明确要求（联网搜索，帮我查一下等）
        input: "联网搜索：co问题的定义"
        {
        "use_search": true,
       
        }
    '''


    response = client.chat.completions.create(
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

    use_search = out.get('use_search')

    if not isinstance(use_search, bool):
        use_search = False

  

    return use_search




# 工具定义
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bocha_websearch",
            "description": "搜索中国相关的内容（如中文新闻、政策、经济数据）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词，尤其是中文或中国相关。"},
                    "freshness": {"type": "string","default": 'noLimit', "description": "搜索时间范围:oneDay、oneMonth、oneYear、noLimit、noLimit"},
                    "count": {"type": "integer", "default": 20, "description": "返回结果数量。"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exa_search",
            "description": "搜索国际内容（如科技、学术论文、全球新闻）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词，适用于非中国主题。"},
                    "num_results": {"type": "integer", "default": 10, "description": "返回结果数量。"},
                    "category": {"type": "string", "default": 'news', "description": "搜索类别：company、research_paper、news、pdf、github、tweet、personal_site、linkedin_profile、financial_report"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "jina_ai_search",
            "description": "使用 Jina AI 搜索多语言内容，arxiv论文和涉及多个国家或者需要交叉验证的任务。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词。"},
               
                },
                "required": ["query"],
            },
        },
    }
]


def exa_search(query,category,num_results) -> Dict[str, Any]:
    """使用Exa搜索引擎"""
    return exa.search_and_contents(
        query=query,
        type="auto",
        category = category,
        num_results=num_results,
        livecrawl="always",
        summary=True,
        highlights=True,
    )


def bocha_websearch(query, count=10,freshness='noLimit') -> str:
    """使用Bocha搜索引擎"""
    headers = {
        'Authorization': 'sk-d6b8a767f5864872bd584665f6bf659b',
        'Content-Type': 'application/json'
    }
    data = {
        "query": query,
        "freshness": freshness,
        "summary": True,
        "count": count
    }
    response = requests.post('https://api.bochaai.com/v1/web-search', headers=headers, json=data)

    if response.status_code == 200:
        json_response = response.json()
        if json_response["code"] != 200 or not json_response["data"]:
            return f"搜索失败: {json_response.get('msg', '未知错误')}"

        webpages = json_response["data"]["webPages"]["value"]
        if not webpages:
            return "未找到相关结果。"

        return "\n\n".join(
            f"结果 {i}:\n标题: {p['name']}\nURL: {p['url']}\n摘要: {p['summary']}"
            for i, p in enumerate(webpages, 1)
        )
    else:
        return f"搜索请求失败: {response.status_code} - {response.text}"

def jina_ai_search(query: str) -> str:
    """调用 Jina AI 搜索 API"""
    
    params = {
        "q": query,
      
    }
    
    response = requests.get(url = "https://s.jina.ai/", params=params, headers=jina_headers)
    if response.status_code == 200:
        results = response.json()['data']
        if not results:
            return "未找到相关结果。"
        return "\n\n".join(
            f"结果 {i}:\n标题: {r['title']}\nURL: {r['url']}\n描述: {r['description']}"
            for i, r in enumerate(results, 1)
        )
    else:
        return f"请求失败: {response.status_code} - {response.text}"

        
def process_tool_calls(tool_calls, messages):
    """处理工具调用并更新消息列表"""
    for call in tool_calls:
        # 从字典中提取函数名和参数
        func_name = call["function"]["name"]
        args = json.loads(call["function"]["arguments"])

        if func_name == "exa_search":
            result = exa_search(args["query"],args.get("category",'news'),args["num_results"])
            content = str(result)
        elif func_name == "bocha_websearch":
            result = bocha_websearch(args["query"], args.get("count"), args.get("freshness",'noLimit'))
            content = result
        elif func_name == "jina_ai_search":
            result = jina_ai_search(args["query"])
            content = result

        messages.append({
            "role": "tool",
            "content": content,
            "tool_call_id": call["id"],
            "name": func_name
        })
    return messages



def main_search(client,model,messages):

    messages.insert(-1, {"role": "system", "content": 
                         "必须通过调用工具来开展联网搜索，按照问题所涉及的范围挑选搜索引擎，中国内容采用bocha_websearch搜索，国际内容则采用exa_search搜索，arxiv论文、多语言和涉及交叉验证的任务采用jina_ai_search。复杂需要交叉验证的复杂任务同时调用多个工具。在回答问题的时候必须给出参考资料的来源，给出url"})

    # 第一步：流式获取工具调用或直接回答
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        parallel_tool_calls=True,
        stream=True,
    )

    # 累积工具调用参数或直接回答内容
    tool_calls = {}

    for chunk in stream:
        if chunk.choices[0].delta.tool_calls:
            for tool in chunk.choices[0].delta.tool_calls:
                idx = tool.index
                if idx not in tool_calls:
                    tool_calls[idx] = {
                        "id": tool.id,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                        "index": idx,
                    }
                if tool.function.name:
                    tool_calls[idx]["function"]["name"] = tool.function.name
                if tool.function.arguments:
                    tool_calls[idx]["function"]["arguments"] += tool.function.arguments
        # elif chunk.choices[0].delta.content:
        #     print(chunk.choices[0].delta.content, end="", flush=True)

    print('tool_calls:',tool_calls)
    if tool_calls:
    
        # 有工具调用
        calls_list = [v for k, v in sorted(tool_calls.items())]
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": calls_list,
        })
        messages = process_tool_calls(calls_list, messages)

        # 流式输出最终回答
        final_stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        # print("\n最终回答:")
        # for chunk in final_stream:
        #     if chunk.choices[0].delta.content:
        #         print(chunk.choices[0].delta.content, end="", flush=True)

        return final_stream
    else:
        messages = messages.pop(-2)

        stream = client.chat.completions.create(
                                                model=model,
                                                messages=messages,
                                                stream=True,
                                            )

        return stream

if __name__ == "__main__":
    main_search()