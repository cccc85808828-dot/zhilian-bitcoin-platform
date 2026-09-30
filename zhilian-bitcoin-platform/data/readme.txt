本提交包不携带完整Elliptic++原始数据及处理中间文件。

数据公开来源、许可与引用要求见项目根目录“数据来源与使用说明.txt”。下载、预处理和完整性检查可依次运行：

powershell -ExecutionPolicy Bypass -File scripts\download_ellipticpp.ps1
python scripts\preprocess_ellipticpp.py --config configs\ellipticpp.yaml
python scripts\verify_ellipticpp.py

离线演示所需的近期比特币主网公开快照已经保存在demo\data\offline_fixtures.json，不需要下载额外数据。
