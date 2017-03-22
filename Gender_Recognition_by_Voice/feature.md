# 从音频文件提取特征的步骤



1. 安装R。windows操作系统安装包的[链接](https://cran.r-project.org/bin/windows/base/) ，其它操作系统的安装包请自行搜索。
2. 切换当前路径为脚本所在路径

点击 文件 > 改变工作目录

3. 在当前工作目录新建名为test的文件夹，将wav格式的音频文件置于其中。提取的特征将会保存在`test.csv`文件中。第一次运行脚本会自动安装依赖的包。如有疑问可以在[这里留言](http://www.cnblogs.com/meelo/p/6582721.html)。

点击 文件 > 运行R脚本文件 > 选择feature.R文件