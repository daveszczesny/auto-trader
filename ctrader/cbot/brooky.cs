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

        private bool warmedUp = false;

        protected override void OnStart()
        {
            Print("Welcome to BrookyAI - Created by Dave Szczesny");
            this.warmup();
        }

        void warmup()
        {
            // initialize the indicators
            ema21 = Indicators.ExponentialMovingAverage(Bars.ClosePrices, 21);
            ema50 = Indicators.ExponentialMovingAverage(Bars.ClosePrices, 50);
            ema200 = Indicators.ExponentialMovingAverage(Bars.ClosePrices, 200);
            
            // initialize variables
            const int warmUpPeriod = 200;
            var closePrices = new double[warmUpPeriod];
            var highPrices = new double[warmUpPeriod];
            var lowPrices = new double[warmUpPeriod];
            var ema21Values = new double[warmUpPeriod];
            var ema50Values = new double[warmUpPeriod];
            var ema200Values = new double[warmUpPeriod];


            // initialize the warmup period values
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

            // form payload
            var payload = new 
            {
                balance = Account.Balance,
                current_prices = closePrices,
                current_highs = highPrices,
                current_lows = lowPrices,
                indicators = new[]
                {
                    new {name = "ema_200", value = ema200Values},
                    new {name = "ema_50", value = ema50Values},
                    new {name = "ema_21", value = ema21Values}
                }
            };

            string jsonData = JsonSerializer.Serialize(payload);

            SendPostRequestToWarmUp(jsonData);
        }

        protected override void OnBar()
        {
            
            if (!this.warmedUp) {
                Print("Bot not warmed up! Restarting warmup sequence.");
                return;
            }

            // Retrive current market data
            double currentBidPrice = Bars.ClosePrices.LastValue;
            double currentHigh = Bars.HighPrices.LastValue;
            double currentLow = Bars.LowPrices.LastValue;
            double currentEma21 = ema21.Result.LastValue;
            double currentEma50 = ema50.Result.LastValue;
            double currentEma200 = ema200.Result.LastValue;

            // Send the data over to BrooksAI
            var payload = new
            {
                balance = Account.Balance,
                unrealized_pnl = Account.UnrealizedGrossProfit,
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
                    string content_ = await response.Content.ReadAsStringAsync();
                    HandleResponse(content_);
                }
                else if(response.StatusCode == System.Net.HttpStatusCode.ServiceUnavailable){
                    Print("Warmup failed to uphold. Restarting warmup. Status Code: " + response.StatusCode);
                    this.warmup();
                }
                else
                {
                    Print("Failed to send data due to unhandled response code.");
                    Print("Status code: " + response.StatusCode);
                }
            }
            catch(Exception e) {
                Print("Failed sending post request");
                Print(e.Message);
            }
        }


        private async void SendPostRequestToWarmUp(string jsonData)
        {
            try
            {
                Print("Attempting to send warmup post request");
                var API = BASE_URL + "/brooksai/warmup";
                var content = new StringContent(jsonData, Encoding.UTF8, "application/json");
                HttpResponseMessage response = await httpClient.PostAsync(API, content);

                if (response.IsSuccessStatusCode)
                {
                    this.warmedUp = true;
                }
                else
                {
                    Print("Failed to send data");
                    Print("Status code: " + response.StatusCode);
                }
            }
            catch (Exception e) {
                Print(e.Message);
            }
        }

        private void HandleResponse(string content)
        {
            if (content == null) {
                Print("No response from server");
                return;
            };

            var response = JsonSerializer.Deserialize<Dictionary<string, string>>(content);

            response.TryGetValue("action", out string action);
            // response.TryGetValue("lot_size", out string lotSize);
            response.TryGetValue("take_profit", out string takeProfit);
            response.TryGetValue("stop_loss", out string stopLoss);

            // convert lot size to volume units
            // double volume = double.Parse(lotSize) * 100_000;
            string symbol = "EURUSD";

            if (action == "BUY" || action == "SELL")
            {
                Print("Entering into trade!");
                TradeType tradeType = action == "BUY" ? TradeType.Buy : TradeType.Sell;
                ExecuteMarketOrder(tradeType, symbol, 1);
            }
            else if (action == "CLOSE")
            {
                Print("Attempting to exit trade!");
                var position = Positions.Find("Brooky");
                if (position != null)
                {
                    ClosePosition(position);
                }
            } else {
                Print("Will do nothing for a minute. Just wait and see what I do :)");
            }

        }
    }
}