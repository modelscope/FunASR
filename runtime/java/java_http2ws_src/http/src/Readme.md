dependencies {
  implementation("org.springframework.boot:spring-boot-starter-web")
  implementation("org.json:json:20240303")
  implementation("org.springframework.boot:spring-boot-starter-websocket")
}


使用接口测试工具 form-data格式传入文件 返回测试成功即运行成功

默认访问路径:
  io路径: http://localhost:8081/recognition/testIO
  nio路径: http://localhost:8081/recognition/testNIO

application.yml中可根据自身需要修改对应模型参数