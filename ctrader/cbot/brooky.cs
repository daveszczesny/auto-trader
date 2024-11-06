using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Linq;
using System.Text;
using System.Text.Json;
using cAlgo.API;
using cAlgo.API.Collections;
using cAlgo.API.Indicators;
using cAlgo.API.Internals;


namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.FullAccess)]
    public class Brooky : Robot
    {
        public int Volume {get; set;}
        private ExponentialMovingAverage ema21;
        private ExponentialMovingAverage ema50;
        private ExponentialMovingAverage ema200;
        
        private readonly HttpClient httpClient = new();
        private readonly string BASE_URL = "https://brooky-api-550951781970.europe-west2.run.app";

        protected override void OnStart()
        {

            // initialize the indicators
            ema21 = Indicators.ExponentialMovingAverage(Bars.ClosePrices, 21);
            ema50 = Indicators.ExponentialMovingAverage(Bars.ClosePrices, 50);
            ema200 = Indicators.ExponentialMovingAverage(Bars.ClosePrices, 200);
            
            int warmUpPeriod = 200;
            var closePrices = new double[warmUpPeriod];
            var highPrices = new double[warmUpPeriod];
            var lowPrices = new double[warmUpPeriod];
            var ema21Values = new double[warmUpPeriod];
            var ema50Values = new double[warmUpPeriod];
            var ema200Values = new double[warmUpPeriod];


            for (int i = 0; i < warmUpPeriod; i++)
            {
                int index = Bars.Count - 1 - i;
                closePrices[i] = Bars.ClosePrices[index];
                highPrices[i] = Bars.HighPrices[index];
                lowPrices[i] = Bars.LowPrices[index];
                ema21Values[i] = ema21.Result[index];
                ema50Values[i] = ema50.Result[index];
                ema200Values[i] = ema200.Result[index];
            }

            var payload = new 
            {
                closePrices = closePrices,
                highPrices = highPrices,
                lowPrices = lowPrices,
                ema21 = ema21Values,
                ema50 = ema50Values,
                ema200 = ema200Values
            }

            string jsonData = JsonSerializer.Serialize(payload);
            SendPostRequestToWarmUp(jsonData);

        }

        protected override void OnBar()
        {
            // Retrive current market data
            DataSeries currentBidPrice = Bars.ClosePrices;
            DataSeries currentHigh = Bars.HighPrices;
            DataSeries currentLow = Bars.LowPrices;
            double currentEma21 = ema21.Result.LastValue;
            double currentEma50 = ema50.Result.LastValue;
            double currentEma200 = ema200.Result.LastValue;

            // Send the data over to BrooksAI
            var payload = new
            {
                balance = Account.Balance,
                unrealized_pnl = Account.UnrealizedPnL,
                current_price = currentBidPrice,
                current_high = currentHigh,
                current_low = currentLow,
                open_trades = 0,
                indicators = new[]
                {
                    new {name = "ema_200", value = currentEma200},
                    new {name = "ema_50", value = currentEma50},
                    new {name = "ema_21", value = currentEma21}
                }
            };

            string jsonData = JsonSerializer.Serialize(payload);

            // Send data over API
            SendPostRequest(jsonData);
        }


        private async void SendPostRequest(string jsonData)
        {
            try
            {
                var API = BASE_URL + "/brooksai/predict";
                var content = new StringContent(jsonData, Encoding.UTF8, "application/json");
                HttpResponseMessage response = await httpClient.PostAsync(API, content);

                if (response.IsSuccessStatusCode)
                {
                    Print("Data sent successfully");

                    string conent = await response.Content.ReadAsStringAsync();
                    HandleResponse(content.ToString());
                }
                else
                {
                    Print("Failed to send data");
                }
            }catch(Exception e){
                Console.WriteLine(e);
            }
        }


        private async void SendPostRequestToWarmUp(string jsonData)
        {
            try
            {
                var API = BASE_URL + "/brooksai/warmup";
                var content = new StringContent(jsonData, Encoding.UTF8, "application/json");
                HttpResponseMessage response = await httpClient.PostAsync(API, content);

                if (response.IsSuccessStatusCode)
                {
                    Print("Data sent successfully");
                }
                else
                {
                    Print("Failed to send data");
                }
            }
        }

        private void HandleResponse(string content)
        {
            if (content == null) return;

            var response = JsonSerializer.Deserialize<Dictionary<string, string>>(content);

            response.TryGetValue("action", out string action);
            response.TryGetValue("lot_size", out string lotSize);
            response.TryGetValue("take_profit", out string takeProfit);
            response.TryGetValue("stop_loss", out string stopLoss);

            // convert lot size to volume units
            double volume = double.Parse(lotSize) * 100_000;
            string symbol = "EURUSD";

            if (action == "BUY" || action == "SELL")
            {
                TradeType tradeType = action == "BUY" ? TradeType.Buy : TradeType.Sell;
                ExecuteMarketOrder(tradeType, symbol, volume);
            }
            else if (action == "CLOSE")
            {

                var position = Positions.Find("Brooky");
                if (position != null)
                {
                    ClosePosition(position);
                }
            }

        }
    }
}