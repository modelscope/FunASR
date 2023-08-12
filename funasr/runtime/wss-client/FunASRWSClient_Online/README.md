# cshape-client-offline

这是一个基于FunASR-Websocket服务器的CShape客户端，用于实时语音识别和转录本地音频文件。

将配置文件放在与程序相同目录下的config文件夹中，并在config.ini中配置服务器ip地址和端口号。

配置好服务端ip和端口号，在vs中打开需添加NAudio和Websocket.Client的Nuget程序包后，可直接进行测试，按照控制台提示操作即可。

注：实时语音识别使用online或2pass，转录文件默认使用offline，在win11下完成测试。