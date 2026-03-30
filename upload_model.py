from modelscope.hub.api import HubApi

api = HubApi()
api.login('ms-7c80103d-ce74-4d4c-932d-256e6b6dc645')  # 或用环境变量 MODELSCOPE_API_TOKEN

# api.push_model(
#     model_id='forgeX/yolo-Q4',
#     model_dir='./runs/detect/runs/detect/road_damage_v1_yolo26l-freeze',         # 本地目录，需包含 configuration.json
#     visibility=5,                           # 1=私有, 5=公开
#     license='Apache License 2.0',
#     commit_message='upload training files',
#     tag='v0.1.0',                           # 可选版本标签
# )

api.upload_folder(
    repo_id='forgeX/yolo-Q4',
    folder_path='./runs/detect/runs/detect/road_damage_v1_yolo26n-aug',
    path_in_repo='yolo26n-200epochs-aug',          # ← 上传到仓库中的子目录名
    commit_message='add yolo26n-200epochs-aug training files',
)

# # 上传文件
# api.upload_file(
#     path_or_fileobj='./runs/detect/runs/detect/road_damage_v1_yolo26l-freeze/result.zip',  # 本地文件路径
#     path_in_repo='yolo26l-freeze-300-epochs',      # 仓库中的路径
#     repo_id='forgeX/yolo-Q4',     # 数据集 ID
#     repo_type='model'                   # 仓库类型：dataset 或 model
# )