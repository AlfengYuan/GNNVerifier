import json
import click
import os
import asyncio
import time
import sys
from openai import OpenAI 

sys.path.append("../")
from utils import get_cur_time

_api_key = os.environ.get("OPENAI_API_KEY")
_api_base_url = os.environ.get("OPENAI_API_BASE")
client = OpenAI(api_key=_api_key, base_url=_api_base_url)

async def inference_one_case(input, temperature, top_p, tool_string, write_file, llm, demo_string, resource_type=False):
    user_request = input["user_request"]
    
    if resource_type:
        prompt = """\n# GOAL #: Based on the above tools, I want you generate task steps and task nodes to solve the # USER REQUEST #. The format must in a strict JSON format, like: {"task_steps": [ step description of one or more steps ], "task_nodes": [{"task": "tool name must be from # TASK LIST #", "arguments": [ a concise list of arguments for the tool. Either original text, or user-mentioned filename, or tag '<node-j>' (start from 0) to refer to the output of the j-th node. ]}], "task_links": [{"source": "task name i", "target": "task name j"}]} """
        prompt += """\n\n# REQUIREMENTS #: \n1. the generated task steps and task nodes can resolve the given user request # USER REQUEST # perfectly. Task name must be selected from # TASK LIST #; \n2. the task steps should strictly aligned with the task nodes, and the number of task steps should be same with the task nodes; \n3. the dependencies among task steps should align with the argument dependencies of the task nodes; \n4. the tool arguments should be align with the input-type field of # TASK LIST #;"""
    else:
        prompt = """\n# GOAL #: Based on the above tools, I want you generate task steps and task nodes to solve the # USER REQUEST #. The format must in a strict JSON format, like: {"task_steps": [ "concrete steps, format as Step x: Call xxx tool with xxx: 'xxx' and xxx: 'xxx'" ], "task_nodes": [{"task": "task name must be from # TASK LIST #", "arguments": [ {"name": "parameter name", "value": "parameter value, either user-specified text or the specific name of the tool whose result is required by this node"} ]}], "task_links": [{"source": "task name i", "target": "task name j"}]}"""
        prompt += """\n\n# REQUIREMENTS #: \n1. the generated task steps and task nodes can resolve the given user request # USER REQUEST # perfectly. Task name must be selected from # TASK LIST #; \n2. the task steps should strictly aligned with the task nodes, and the number of task steps should be same with the task nodes; \n3. The task links (task_links) should reflect the temporal dependencies among task nodes, i.e. the order in which the APIs are invoked;"""

    prompt += demo_string
    prompt += """\n\n# USER REQUEST #: {{user_request}}\nnow please generate your result in a strict JSON format:\n# RESULT #:"""
    final_prompt = tool_string + prompt.replace("{{user_request}}", user_request)

    payload = {
        "model": llm,  
        "messages": [{"role": "user", "content": final_prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": 2000
    }
    st_time = time.time()

    try:
        returned_content = await get_response(payload, resource_type)
    except Exception as e:
        print(f"Failed #id {input['id']}: {type(e)} {e}")
        raise e 
    
    res = {
        "id": input["id"],
        "user_request": input["user_request"],
        "task_steps": returned_content["task_steps"],
        "task_nodes": returned_content["task_nodes"],
        "task_links": returned_content["task_links"],
        "cost_time": round(time.time() - st_time, 4)
    }
        
    write_file.write(json.dumps(res) + "\n")
    write_file.flush()


async def get_response(payload, resource_type=False):
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(**payload))
        
        content = response.choices[0].message.content
        content = content.replace("\n", "").replace("\_", "_").replace("\\", "")
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return await handle_json_error(content, payload, resource_type)
            
    except Exception as e:
        raise Exception(f"API Error: {str(e)}")


async def handle_json_error(origin_content, original_payload, resource_type):
    if resource_type:
        prompt = """Please format the result # RESULT # to a strict JSON format # STRICT JSON FORMAT #. \nRequirements:\n1. Do not change the meaning of task steps, task nodes and task links;\n2. Don't tolerate any possible irregular formatting to ensure that the generated content can be converted by json.loads();\n3. You must output the result in this schema: {"task_steps": [ step description of one or more steps ], "task_nodes": [{"task": "tool name must be from # TASK LIST #", "arguments": [ a concise list of arguments for the tool. Either original text, or user-mentioned filename, or tag '<node-j>' (start from 0) to refer to the output of the j-th node. ]}], "task_links": [{"source": "task name i", "target": "task name j"}]}\n# RESULT #:{{illegal_result}}\n# STRICT JSON FORMAT #:"""
    else:
        prompt = """Please format the result # RESULT # to a strict JSON format # STRICT JSON FORMAT #. \nRequirements:\n1. Do not change the meaning of task steps, task nodes and task links;\n2. Don't tolerate any possible irregular formatting to ensure that the generated content can be converted by json.loads();\n3. Pay attention to the matching of brackets. Write in a compact format and avoid using too many space formatting controls;\n4. You must output the result in this schema: {"task_steps": [ "concrete steps, format as Step x: Call xxx tool with xxx: 'xxx' and xxx: 'xxx'" ], "task_nodes": [{"task": "task name must be from # TASK LIST #", "arguments": [ {"name": "parameter name", "value": "parameter value, either user-specified text or the specific name of the tool whose result is required by this node"} ]}], "task_links": [{"source": "task name i", "target": "task name j"}]}\n# RESULT #:{{illegal_result}}\n# STRICT JSON FORMAT #:"""

    prompt = prompt.replace("{{illegal_result}}", origin_content)
    new_payload = original_payload.copy()
    new_payload["messages"] = [{"role": "user", "content": prompt}]
    
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(**new_payload)
        )
        
        content = response.choices[0].message.content
        content = content.replace("\n", "").replace("\_", "_")
        start_pos = content.find("STRICT JSON FORMAT #:")
        if start_pos != -1:
            content = content[start_pos + len("STRICT JSON FORMAT #:"):]

        content = content[content.find("{"):content.rfind("}") + 1]
        return json.loads(content)
    except Exception as e:
        raise Exception(f"Retry failed: {str(e)}")


@click.command()
@click.option("--dataset", default="huggingface")
@click.option("--temperature", type=float, default=0.2)
@click.option("--top_p", type=float, default=0.1)
@click.option("--api_addr") 
@click.option("--api_port", type=int, default=443) 
@click.option("--llm", type=str, default="gpt-3.5-turbo")  
@click.option("--use_demos", type=int, default=1)
@click.option("--multiworker", type=int, default=4)
def main(dataset, temperature, top_p, api_addr, api_port, llm, use_demos, multiworker):
    print('= ' * 20)
    print('## Starting Time:', get_cur_time(), flush=True)

    llm_short_names = {
        "gpt-3.5-turbo": "GPT-3.5",
        "gpt-4o": "GPT-4O"
    }
    llm_short = llm_short_names.get(llm, llm)
    prediction_dir = f"../prediction/{dataset}/{llm_short}"

    infer_step_file = f"{prediction_dir}/direct.json"

    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)
    
    alignment = json.load(open(f"../data/{dataset}/split_ids.json", 'r'))["test_ids"]
    alignment_ids = alignment["chain"]
    
    has_inferenced = set()
    if os.path.exists(infer_step_file):
        with open(infer_step_file, 'r') as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "id" in data:
                    has_inferenced.add(data["id"])
    
    user_request_file = open(f"../data/{dataset}/user_requests.json", 'r')
    inputs = []
    for line in user_request_file:
        input = json.loads(line)
        if input["id"] not in alignment_ids:
            continue
        if input["id"] not in has_inferenced:
            inputs.append(input)
    user_request_file.close()

    write_file = open(infer_step_file, "a") 
    print(infer_step_file)
    
    
    tool_list = json.load(open(f"../data/{dataset}/tool_desc.json", "r"))["nodes"]
    tool_string = "# TASK LIST #:\n"
    for k, tool in enumerate(tool_list):
        tool_string += json.dumps(tool) + "\n"

    demo_string = ""
    if use_demos:
        demos_id_list = {
            "huggingface": ["10523150", "14611002", "22067492"],
            "multimedia": ["30934207", "20566230", "19003517"],
            "dailylife": ["27267145", "91005535", "38563456"],
            "tmdb": [1],
            "ultratool": ["3010"]
        }

        demos_id = demos_id_list[dataset][:use_demos]

        demos_rf = open(f"../data/{dataset}/data.json", "r")
        demos = []
        for line in demos_rf:
            data = json.loads(line)
            if data["id"] in demos_id:
                demo = {
                    "user_request": data["user_request"],
                    "result": {
                        "task_steps": data["task_steps"],
                        "task_nodes": data["task_nodes"],
                        "task_links": data["task_links"]
                    }
                }
                demos.append(demo)
        demos_rf.close()
    
        if len(demos) > 0:
            demo_string += "\nHere are provided examples for your reference.\n"
            for demo in demos:
                demo_string += f"""\n# EXAMPLE #:\n# USER REQUEST #: {demo["user_request"]}\n# RESULT #: {json.dumps(demo["result"])}"""

    sem = asyncio.Semaphore(multiworker)
    
    resp_type = dataset in ["huggingface", "multimedia"]
    async def inference_wrapper(input, temperature, top_p, tool_string, write_file, llm, demo_string, resource_type):
        async with sem:
            await inference_one_case(input, temperature, top_p, tool_string, write_file, llm, demo_string, resource_type)
    
    if len(inputs) == 0:
        print("All Completed!")
        return 
    else:
        print(f"Detected {len(has_inferenced)} has been inferenced, ")
        print(f"Start inferencing {len(inputs)} tasks ... ")
    
    loop = asyncio.get_event_loop()

    tasks = []
    for input in inputs:
        if input["id"] not in demos_id:
            tasks.append(inference_wrapper(input, temperature, top_p, tool_string, write_file, llm, demo_string, resource_type=resp_type))
        else:
            print(f"Case {input['id']} in {use_demos}-shot examples and thus Skip")
        
    done, failed = [], []
    results = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
    for result in results:
        if isinstance(result, Exception):
            print(result)
            failed.append(result)
        else:
            done.append(result)

    print(f"Completed {len(done)} Failed {len(failed)}")
    loop.close()

    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")
    
        
if __name__ == "__main__":
    main()
