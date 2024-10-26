# BrookyAPI

BrookyAPI is the front of the project, it allows us to access the AI via a RESTful API endpoint.

# BrooksAI via BrookyAPI

## Loading up
BrooksAI uses LSTM states for better predictions. These LSTM states need to be stabilized before use.

### Warm up period
When deploying / "enabling" the AI the AI won't be able to perform any action until it warmed up, and is ready to take on the markets.


# Deploying

1. Build AI from brooksai using tool
2. run the following command from brooksapi/
```bash
gcloud run deploy --source .
```



## TODO when I get better WiFi,

Run the following command from root
```bash
docker build -t gcr.io/auto-trader-439718/brooky-api -f brookyapi/dockerfile .
```