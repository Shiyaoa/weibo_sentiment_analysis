# weibo_sentiment_analysis

sentiment analysis for some texts in China social media sina weibo

the dataset "labeleddata.csv" is a covid-19 weibo dataset
 





使用bert模型进行训练测试，执行命令如下
```
python main.py --model bert
```
基于训练好的bert模型预测新数据，执行命令如下
```
python predict.py --model bert --predict "your sentence"
```
使用ERNIE模型进行训练测试，执行命令如下
```
python main.py --model ERNIE
```
基于训练好的ERNIE模型预测新数据，执行命令如下
```
python predict.py --model ERNIE --predict "your sentence"
