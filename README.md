# SARIMA_model
This is the script what optimizing SARIMA model's parameters and predicting time series data.<br>
Seasonal wave assume as weekly change of stock price.<br>
New York Stock Exchange's cord is first argument and predicting duration is second.<br>
Results contains Raw Data, Seasonal Decompose, Modeling results and Forecast of data.<br>
Excecute like script below.
This is Microsoft's 5days forecasting of stock price.<br>
Developer assumes no responsibility to results.<br>
```
python tsa_predict.py MSFT 5
```
SARIMA modelのパラメータを最適化して，時系列データの予測をするスクリプトです．<br>
季節性は週ごとの変化を仮定しています．<br>
NY市場のコードを第一引数，予測したい日数を第二引数にすると，実行できます．<br>
出力結果に，Raw Data, Seasonal Decompose, 学習結果, 予測値が含まれています．<br>
上記のスクリプトはマイクロソフト社の株価の5日間予測です．<br>
予測日数が長期になるほど予測は不正確です．<br>
予測の結果に関しては一切の責任を追いません．<br>
