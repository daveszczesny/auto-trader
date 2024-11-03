# BrookyAPI

BrookyAPI is the front of the project, it allows us to access the AI via a RESTful API endpoint.

# BrooksAI via BrookyAPI

## Loading up
BrooksAI uses LSTM states for better predictions. These LSTM states need to be stabilized before use.

### Warm up period
When deploying / "enabling" the AI the AI won't be able to perform any action until it warmed up, and is ready to take on the markets.


# Usage
Using `postman` we can call our api endpoint `https://brooky-api-550951781970.europe-west2.run.app/brooksai [POST]`
```
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "balance": {
      "type": "number",
      "description": "The balance of the account."
    },
    "unrealized_pnl": {
      "type": "number",
      "description": "The unrealized profit and loss."
    },
    "current_price": {
      "type": "number",
      "description": "The current price."
    },
    "current_high": {
      "type": "number",
      "description": "The current high price."
    },
    "current_low": {
      "type": "number",
      "description": "The current low price."
    },
    "open_trades": {
      "type": "integer",
      "description": "The number of open trades."
    },
    "indicators": {
      "type": "array",
      "description": "List of indicators.",
      "items": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "description": "The name of the indicator."
          },
          "value": {
            "type": "number",
            "description": "The value of the indicator."
          }
        },
        "required": ["name", "value"]
      }
    }
  },
  "required": [
    "balance",
    "unrealized_pnl",
    "current_price",
    "current_high",
    "current_low",
    "open_trades",
    "indicators"
  ]
}
```

The currently expects ema 200, ema 50, and ema 21 as part of its payload, and must be included.