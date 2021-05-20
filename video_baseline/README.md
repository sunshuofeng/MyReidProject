要运行代码，首先
下载https://github.com/liangzheng06/MARS-evaluation 里的info文件夹
``` 
cd datasets

python create_MARS_database.py --data_dir your mars path --info_dir your mars info path
```
然后会在里面生成读取mars需要的文件

然后再运行main文件

``` python
python main.py
```
