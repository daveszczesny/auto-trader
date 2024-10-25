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

        protected override void OnStart()
        {

            // initialize the indicators
            ema21 = Indicators.ExponentialMovingAverage(Bars.ClosePrices, 21);
            ema50 = Indicators.ExponentialMovingAverage(Bars.ClosePrices, 50);
            ema200 = Indicators.ExponentialMovingAverage(Bars.ClosePrices, 200);
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

            var data = new
            {
                BidPrice = currentBidPrice,
                High = currentHigh,
                Low = currentLow,
                Ema21 = currentEma21,
                Ema50 = currentEma50,
                Ema200 = currentEma200
            };

            string jsonData = JsonSerializer.Serialize(data);

            // Send data over API
            SendPostRequest(jsonData);
        }


        private async void SendPostRequest(string jsonData)
        {
            try
            {
                var API = "{API-ENDPOINT}";
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