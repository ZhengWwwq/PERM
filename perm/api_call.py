import asyncio
import random
from itertools import cycle

from openai import AsyncOpenAI


async def get_completion_async(
    client: AsyncOpenAI,
    model_name: str,
    prompt,
    return_reasoning: bool,
    max_retry: int,
    temperature: float,
    top_p: float,
    max_completion_tokens: int,
    semaphore: asyncio.Semaphore,
):
    async with semaphore:
        backoff = 1.5
        is_gpt5 = model_name.startswith("gpt-5")

        for attempt in range(1, max_retry + 1):
            try:
                if is_gpt5:
                    resp = await client.responses.create(
                        model=model_name,
                        input=prompt,
                        max_output_tokens=max_completion_tokens,
                    )

                    output_text = resp.output_text or ""
                    assert len(output_text) > 10

                    if return_reasoning:
                        reasoning_text = ""
                        rc = getattr(resp, "reasoning_content", None)
                        if rc:
                            reasoning_text = "".join(
                                getattr(block, "text", "") for block in rc
                            )
                        return output_text, reasoning_text
                    else:
                        return output_text

                else:
                    resp = await client.chat.completions.create(
                        model=model_name,
                        messages=prompt,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_completion_tokens,
                    )

                    content = resp.choices[0].message.content or ""

                    if return_reasoning:
                        return content, None
                    else:
                        return content

            except Exception as e:
                if attempt == max_retry:
                    continue
                # print(f"[{model_name}] Error on attempt {attempt}: {e}")
                await asyncio.sleep(backoff * (2 ** (attempt - 1)) + random.random())


async def call_with_index(index: int, **kwargs):
    try:
        result = await get_completion_async(**kwargs)
        return index, result
    except Exception:
        return index, None


async def api_call_async(
        model_name: str, 
        user_prompt_list, 
        api_list: list[str], 
        base_url_list: list[str],
        api_call_limit: int,
        max_retry: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_completion_tokens: int = 512):

    deepseek_model_names = ["deepseek-chat", "deepseek-reasoner"]
    openai_model_names = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-5-mini", "gemini-2.5-pro", "gpt-5.1", "gpt-5"]
    # supported_model_names = deepseek_model_names + openai_model_names
    # if model_name not in supported_model_names:
    #     raise ValueError(f"{model_name} must be of the following names: {', '.join(supported_model_names)}")
    return_reasoning = model_name == "deepseek-reasoner"
    
    
    # 1. 准备创建客户端所需的参数列表
    client_params_list = []
    if model_name in deepseek_model_names:
        for api in api_list:
            client_params_list.append({
                "api_key": api,
                "base_url": "https://api.deepseek.com",
            })
    else:
        for api, url in zip(api_list, base_url_list):
            client_params_list.append({
                "api_key": api,
                "base_url": url,
            })

    semaphore_list = [asyncio.Semaphore(api_call_limit) for _ in api_list]

    client_prompts = [[] for _ in api_list]
    for i, user_prompt in enumerate(user_prompt_list):
        client_index = i % len(api_list)
        client_prompts[client_index].append((i, user_prompt))


    async def manage_client_tasks(
        client_params: dict, 
        prompts_with_index: list, 
        semaphore: asyncio.Semaphore
    ):
        async with AsyncOpenAI(**client_params) as client:
            tasks = []
            for index, prompt in prompts_with_index:
                task = asyncio.create_task(call_with_index(
                    index=index,
                    client=client,
                    model_name=model_name,
                    prompt=prompt,
                    return_reasoning=return_reasoning,
                    max_retry=max_retry,
                    temperature=temperature,
                    top_p=top_p,
                    max_completion_tokens=max_completion_tokens,
                    semaphore=semaphore,
                ))
                tasks.append(task)

            if tasks:
                return await asyncio.gather(*tasks)
            return []

    manager_tasks = []
    for params, prompts, semaphore in zip(client_params_list, client_prompts, semaphore_list):
        if prompts:
            manager_task = asyncio.create_task(
                manage_client_tasks(params, prompts, semaphore)
            )
            manager_tasks.append(manager_task)

    all_results_nested = await asyncio.gather(*manager_tasks)

    final_results_with_index = []
    for client_batch_results in all_results_nested:
        final_results_with_index.extend(client_batch_results)

    results = [None] * len(user_prompt_list)
    for index, result in final_results_with_index:
        results[index] = result

    return results


def api_call(
        model_name: str, 
        user_prompt_list, 
        api_list: list[str], 
        base_url_list: list[str],
        api_call_limit: int,
        max_retry: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_completion_tokens: int = 512):
    
    return asyncio.run(api_call_async(model_name=model_name,  
                                      user_prompt_list=user_prompt_list, 
                                      api_list=api_list, 
                                      base_url_list=base_url_list,
                                      api_call_limit=api_call_limit, 
                                      max_retry=max_retry, 
                                      temperature=temperature, 
                                      top_p=top_p, 
                                      max_completion_tokens=max_completion_tokens))