import argparse
import asyncio
import logging
import os
from collections import defaultdict
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    TypeVar,
    cast,
)

import httpx

# import uvicorn
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

parser = argparse.ArgumentParser(
    description="Smart Patrol Inspection MCP Server"
)
parser.add_argument(
    "-p",
    "--port",
    type=int,
    default=8004,
    help="The port to use. Default: `%(default)d`",
)
parser.add_argument(
    "-u",
    "--url",
    type=str,
    default="http://112.126.99.154:8823/bk/siact-sec-api",
    help="The digital twin server address. Default: `%(default)s`",
)
parser.add_argument(
    "-t",
    "--transport",
    type=str,
    default="sse",
    help="The transport you are prefer for. Default: `%(default)s`. Allowed values: `sse`, `streamable-http or http`, `stdio`",
)
parser.add_argument(
    "-l",
    "--log-level",
    type=str,
    default="INFO",
    help="Log level. Default: `%(default)s`",
    dest="log_level",
)
parser.add_argument(
    "--json",
    action="store_true",
    help="Output the result in JSON format. Default: `%(default)s`",
)
args = parser.parse_args()

# Compatible with the abbreviated form of streamble-http - http
if args.transport == "http":
    args.transport = "streamable-http"

# Check the transport type and set the corresponding flags
transport_is_http = args.transport == "streamable-http"


# Configure logging
logging.basicConfig(
    level=getattr(logging, args.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
)
logger = logging.getLogger("[SmartPatrolInspectionServer]")


mcp = FastMCP(
    name="smart patrol inspection",
    instructions="Digital twin integration through Model Context Protocol for smart patrol inspection.",
    port=args.port,
    host="0.0.0.0",
    stateless_http=True if transport_is_http else False,
    json_response=args.json,
)

T = TypeVar("T")


class DTResponse(BaseModel, Generic[T]):
    """数字孪生接口返回的数据结构"""

    code: int
    msg: str
    data: T


async def request_dt(
    route: str,
    *,
    method: Literal["GET", "POST"],
    data: Dict | List,
    params: Any,
    timeout: int = 30,
) -> Dict | str:
    """
    Sends an asynchronous HTTP request to the Digital Twin API.

    Args:
        route (str): The API endpoint route to request.
        method (Literal["GET", "POST"]): The HTTP method to use for the request.
        data (Dict): The request payload for POST requests.
        params (Any): Query parameters for GET requests.
        timeout (int, optional): Request timeout in seconds. Defaults to 30.

    Returns:
        Dict | str: The JSON response if successful, or an error message string.

    Raises:
        ValueError: If the DT_URL environment variable is not set.
    """

    base_url = (
        os.getenv("DT_URL", args.url)
        or "http://112.126.99.154:8823/bk/siact-sec-api"
    )
    if not base_url:
        raise ValueError("DT_URL environment variable not set")

    try:
        async with httpx.AsyncClient(
            base_url=base_url, timeout=timeout
        ) as client:
            logger.debug(f"[Request Url] ==> {base_url + route}")

            if method == "POST":
                logger.debug(f"[Request Data] ==> {data}")
                response = await client.post(route, json=data)

            elif method == "GET":
                if params:
                    logger.debug(f"[Request Params] ==> {params}")
                    response = await client.get(route, params=params)
                else:
                    response = await client.get(route)

            if int(response.status_code) == 200:
                resp = response.json()
                return resp
            else:
                return f"Error: The route {route} not found."

    except Exception as e:
        logger.error(f"[DT] Api request error: {e}")
        return f"Error quering digital twin: {e}"


async def get_all_projects(project_type: Optional[str] = None) -> Dict | str:
    """获取数字孪生的项目实例列表

    Args:
        project_type (str): 项目类别, 可为空.

    Returns:
        (Dict): 返回的数字孪生的项目实例列表的所有数据
    """

    route = "/common"
    if not project_type:
        params = {}
    else:
        params = {"proType": project_type.upper()}

    return await request_dt(
        route, method="GET", data={}, params=params, timeout=30
    )


async def get_eq_insCode_by_project_insCode_and_modelCode(
    project_insCode: str, eqModelCode: str, eqModelName: str
) -> Dict[str, Dict]:
    """根据项目实例码和设备模型码获取所有的设备实例码

    Args:
        project_insCode (str): 项目的实例码
        eqModelCode (str): 设备的模型码
        eqModelName (str): 设备的模型名称
    """

    route = "/common/ins/list"
    method = "GET"
    params: Dict = {"dataCode": project_insCode, "modelDataCode": eqModelCode}
    data: Dict = {}
    response = await request_dt(route, method=method, data=data, params=params)
    if isinstance(response, str):
        return {}
    else:
        return {eqModelName: response}


async def get_model_tree_by_project_model_code(
    project_model_code: str,
) -> Dict | str:
    """根据项目模型码获取所有的设备树信息

    Args:
        project_model_code (str): 项目模型码

    Returns:
        Dict: 数字孪生返回接口信息
    """

    route = "/model/list"
    method = "GET"
    params = {"dataCode": project_model_code, "hasEq": True, "hasSub": True}
    data: Dict = {}
    return await request_dt(route, method=method, data=data, params=params)


def recursive_find_model_code(
    model_tree: List[Dict], model_name: str, model_type: str = "eq"
) -> List[str]:
    """递归查找设备模型码"""

    results: List[str] = []

    for node in model_tree:
        if node.get("nodeType") == model_type and model_name in (
            node.get("modelName", "") or ""
        ):
            results.append(node.get("dataCode", ""))
        else:
            if children := node.get("children"):
                results.extend(
                    recursive_find_model_code(children, model_name, model_type)
                )

    return results


async def get_eq_model_code(
    project_model_code: str, eq_name: str
) -> Dict[str, List[str]]:
    """获取设备模型码"""

    resp = await get_model_tree_by_project_model_code(project_model_code)
    if isinstance(resp, str):
        return {}

    model_tree = resp.get("data", [])

    # 通过"nodeType": "eq"和"modelName": "xxx"来一起匹配
    return {eq_name: recursive_find_model_code(model_tree, eq_name)}


def filter_dict(data: Dict | List, keys: List[str]) -> Dict | List:
    """过滤字典中的键"""

    if isinstance(data, list):
        return [
            filter_dict(item, keys) if isinstance(item, dict) else item
            for item in data
        ]
    else:
        return {k: v for k, v in data.items() if k not in keys}


# @mcp.tool(name="get_instances_by_model_code")
async def get_instances_in_dt(
    project_name: str = Field(description="项目名称"),
    model_names: List[str] = Field(
        description="[系统/站/单元/设备]类别名称列表"
    ),
):
    """根据项目名称和设备实例名称列表获取设备实例信息

    Args:
        project_name (str, optional): 项目名称. Defaults to Field(description="项目名称").
        model_names (List[str], optional): 设备实例名称列表. Defaults to Field(description="[系统/站/单元/设备]类别名称列表").

    Returns:
        List[Dict | List | str] | str: 设备实例信息列表或者错误信息.
    """

    all_projects = await get_all_projects(None)

    if isinstance(all_projects, str):
        return []

    data = all_projects["data"]
    project_insCode: str = ""
    project_model_code: str = ""
    for project in data:
        if project["proName"] == project_name:
            project_insCode = project.get("dataCode", "") or ""
            project_model_code = project.get("modelDataCode", "") or ""
            break

    # 并发获取项目下所有设备实例的insCode, 需要解构
    tasks = [
        get_eq_model_code(project_model_code, eq_name)
        for eq_name in model_names
    ]
    # [{模型名称: [模型码1, 模型码2, ...]}]
    eq_name_model_codes_ = await asyncio.gather(*tasks)
    # 对模型码进行去重
    eq_model_codes: Dict[str, str] = {}
    for subitem in eq_name_model_codes_:
        # 模型名称:[模型码, ...]
        for eq_model_name, model_codes_ in subitem.items():
            for model_code in model_codes_:
                eq_model_codes[model_code] = eq_model_name

    results = await asyncio.gather(
        *[
            get_eq_insCode_by_project_insCode_and_modelCode(
                project_insCode, eq_model_code, eq_model_name
            )
            for eq_model_code, eq_model_name in eq_model_codes.items()
        ]
    )
    return [
        {
            eq_model_name: filter_dict(
                DTResponse[Dict | List](**res).data, ["insId"]
            )
        }
        if isinstance(res, dict)
        else {eq_model_name: res}
        for _res in results
        for eq_model_name, res in _res.items()
    ]


# @mcp.tool(name="get_dynamic_prop_list")
# async def get_dynamic_prop_list(
#     dataCodes: List[str] = Field(description="设备实例的数字化编码列表"),
# ) -> List[Dict | str]:
#     """获取`设备`动态属性列表.

#     Args:
#         dataCodes (List[str]): 设备实例的数字化编码列表, **必填**.

#     Returns:
#         List[Dict | str]: 动态属性列表的数据或错误消息.
#     """

#     route = "/v1/ins/eq/page/dynamic"

#     def _compose_data_code_params(_dataCodes: List[str]) -> List[Dict]:
#         _data: Dict = {
#             "propTypes": [],
#             "propCode": None,
#             "propName": None,
#             "pageNumber": 1,
#             "pageSize": 500,
#         }
#         return [{**_data, "dataCode": dataCode} for dataCode in _dataCodes]

#     data_params = _compose_data_code_params(dataCodes)

#     async def handle_request(data: Dict) -> Dict | str:
#         resp = await request_dt(route, method="POST", data=data, params={})

#         if isinstance(resp, str):
#             return {data["dataCode"]: resp}
#         else:
#             datas = resp["data"]["records"]

#             new_datas = []
#             for d in datas:
#                 new_datas.append(
#                     {
#                         "dataCode": d["dataCode"],
#                         "propName": d["propName"],
#                     }
#                 )

#             return {data["dataCode"]: new_datas}

#     results = await asyncio.gather(
#         *[handle_request(data) for data in data_params]
#     )

#     return results


async def request_ins_dyinfo_batch(ins_data_codes: List[str]) -> Dict | str:
    """批量获取实例设备的属性信息"""

    route = "/common/batch/all"
    data = {"dataCodes": ins_data_codes, "propGroups": ["dynamic"]}

    response = await request_dt(route, method="POST", data=data, params={})
    return response


class PropertyModel(TypedDict):
    dataCode: str
    propName: str


class AgentInspectModel(TypedDict):
    dataCode: str
    device: str
    inspectionContent: List[PropertyModel]


@mcp.tool("get_inspect_contents_by_project_and_eq_model_code_and_prop_names")
async def get_inspect_contents_by_project_and_eq_model_code_and_prop_names(
    project_name: str = Field(description="项目名称"),
    eq_props_names: List[
        Dict[Literal["name", "props"], str | List[str]]
    ] = Field(
        description="设备名称列表和属性列表",
        examples=[
            [{"name": "设备名称", "props": ["属性名称1", "属性名称2", "..."]}]
        ],
    ),
) -> List[AgentInspectModel] | str:
    """根据项目名称、设备类别名称和属性名称获取属性的检验内容.

    Args:
        project_name (str): 项目名称.
        eq_props_names (List[str]): 设备名称列表和属性列表. Eg. `[{name: 设备名称, props: [属性名称1, 属性名称2...]}]`

    Returns:
        List[AgentInspectModel]: 属性的检验内容列表. Eg. `[{dataCode: 设备编码, device: 设备名称, inspectionContent: [{dataCode: 属性编码, propName: 属性名称}]}]`
    """

    # {eq_code: [eq_props1, ...]}
    # 关联设备模型名称与模型实例码
    model_names = cast(List[str], [e["name"] for e in eq_props_names])
    instances = cast(
        List[Dict[str, List[Dict[str, str]] | Dict]],
        await get_instances_in_dt(project_name, model_names),
    )

    eq_datas: Dict[str, str] = {}
    # 归类设备实例名称和模型名称
    eq_ins_code_model_name: Dict[str, str] = {}
    for ins in instances:
        for eq_model_name, _ins in ins.items():
            if isinstance(_ins, dict):
                eq_datas[_ins["dataCode"]] = _ins["insName"]
                eq_ins_code_model_name[_ins["dataCode"]] = eq_model_name
            elif isinstance(_ins, list):
                for item in _ins:
                    if isinstance(item, dict):
                        eq_datas[item["dataCode"]] = item["insName"]
                        eq_ins_code_model_name[item["dataCode"]] = eq_model_name

    prop_resp = await request_ins_dyinfo_batch(list(eq_datas.keys()))

    if isinstance(prop_resp, str):
        return prop_resp
    elif isinstance(prop_resp, dict):
        # 将结果按照设备实例码进行归类
        classify_props: Dict[str, List[PropertyModel]] = defaultdict(list)
        # 防止data为None
        if datas := prop_resp.get("data", []):
            for data in datas:
                if single_datas := data.get("dynamicProperties", []):
                    for s_d in single_datas:
                        if ins_code := s_d.get("insDataCode"):
                            classify_props[ins_code].append(
                                {
                                    "dataCode": s_d["dataCode"],
                                    "propName": s_d["propName"],
                                }
                            )

        # 归类模型名称和属性名称
        model_name_mcp_props_name = {}
        for _eq_prop in eq_props_names:
            model_name_mcp_props_name[_eq_prop["name"]] = _eq_prop["props"]

        new_datas: List[AgentInspectModel] = []
        for eq_code, eq_name in eq_datas.items():
            # 获取到指定设备的属性信息
            eq_code_props = classify_props.get(eq_code, []) or []
            # 设备实例码对应的设备模型名称
            eq_code_map_model_name = (
                eq_ins_code_model_name.get(eq_code, "") or ""
            )
            _inspect_contents: List[PropertyModel] = []

            new_inspect_contents: List[PropertyModel] = []
            # 通过设备实例码获取设备模型名称对应的属性名称列表
            current_eq_code_param_props: List[str] = (
                model_name_mcp_props_name.get(eq_code_map_model_name, [])
            )
            for param_prop_name in current_eq_code_param_props:
                _p = PropertyModel(dataCode="", propName=param_prop_name)
                for prop in eq_code_props:
                    prop_name = prop["propName"]
                    if _p["propName"] == prop_name:
                        _p["dataCode"] = prop["dataCode"]

                new_inspect_contents.append(_p)

            new_datas.append(
                AgentInspectModel(
                    dataCode=eq_code,
                    device=eq_name,
                    inspectionContent=new_inspect_contents,
                )
            )
        return new_datas
    else:
        return []


@mcp.tool("query_realtime_data_batch_by_dynamic")
async def query_realtime_data_batch_by_dynamic(
    data_codes: List[str],
) -> List[Dict] | str:
    """属性实时数据(含公式计算)批量查询(动态属性).

    Args:
        data_codes (List[str]): 动态属性的数字化编码列表.

    Returns:
        List[Dict] | str: 含动态属性的历史数据或错误消息.
    """

    route = "/prop/dy/rt/fm"
    response = await request_dt(
        route, method="POST", params={}, data=data_codes, timeout=30
    )

    if isinstance(response, dict):
        data = response.get("data", [])

        if not data:
            return "没有查询到任何数据"
        else:
            return [
                {
                    "propName": d["propName"],
                    "propVal": d["propVal"],
                    "dataCode": d["dataCode"],
                }
                for d in data
            ]

    return response


def entry():
    """The main entry point for the digital twin server."""

    try:
        logger.info(
            "[Smart Patrol Inspection] server started with transport: {}!".format(
                args.transport
            )
        )
        mcp.run(transport=args.transport)
    except Exception as e:
        logger.error(e)
        raise e


if __name__ == "__main__":
    try:
        entry()
    except KeyboardInterrupt:
        logger.error("[KeyboardInterrupt] - Graceful shutdown...")
    except Exception as e:
        logger.error(e)
    finally:
        logger.info("Bye...")
