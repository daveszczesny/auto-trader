# BrookyAPI

BrookyAPI is the front of the project, it allows us to access the AI via a RESTful API endpoint.

# BrooksAI via BrookyAPI

## Loading up
BrooksAI uses LSTM states for better predictions. These LSTM states need to be stabilized before use.

### Warm up period
When deploying / "enabling" the AI the AI won't be able to perform any action until it warmed up, and is ready to take on the markets.


# Usage
Using `postman` we can call our [POST] api endpoint `https://brooky-api-550951781970.europe-west2.run.app/brooksai`
```
{
    "balance": 1000.0,
    "unrealized_pnl": 0.0,
    "current_price": 1.459,
    "current_high": 1.46,
    "current_low": 1.459,
    "open_trades": 0,
    "indicators": [
        {
            "name": "ema_200",
            "value": 1.459
        },
        {
            "name": "ema_50",
            "value": 1.459
        },
        {
            "name": "ema_21",
            "value": 1.459
        }
    ]
}
```

The currently expects ema 200, ema 50, and ema 21 as part of its payload, and must be included.