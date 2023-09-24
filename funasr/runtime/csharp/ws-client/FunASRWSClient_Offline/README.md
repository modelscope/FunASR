# cshape-client-offline

这是一个基于FunASR-Websocket服务器的CShape客户端，用于转录本地音频文件。

将配置文件放在与程序相同目录下的config文件夹中，并在config.ini中配置服务器ip地址和端口号。

配置好服务端ip和端口号，在vs中打开需添加Websocket.Client的Nuget程序包后，可直接进行测试，按照控制台提示操作即可。

更新：支持热词和时间戳，热词需将config文件夹下的hotword.txt放置在执行路径下。

注：运行后台须注意热词和时间戳为不同模型，本客户端在win11下完成测试，编译环境VS2022。
