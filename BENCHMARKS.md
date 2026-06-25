# WhisperKit Benchmarks

This document describes how to run the benchmarks for WhisperKit. The benchmarks can be run on a specific device or all connected devices. The results are saved in JSON files and can be uploaded to the [argmaxinc/whisperkit-evals-dataset](https://huggingface.co/datasets/argmaxinc/whisperkit-evals-dataset) dataset on HuggingFace as a pull request. Below are the steps to run the benchmarks locally in order to reproduce the results shown in our [WhisperKit Benchmarks](https://huggingface.co/spaces/argmaxinc/whisperkit-benchmarks) space.

## Download the Source

To download the code to run the test suite, run:

```sh
git clone git@github.com:argmaxinc/argmax-oss-swift.git
```

## Local Environment

Before running the benchmarks, you'll need to set up your local environment with the necessary dependencies. To do this, run:

```sh
make setup
```

See [Contributing](CONTRIBUTING.md) for more information.


## Xcode Environment

When running the tests, the model to test needs is provided to the Xcode from Fastlane as an environment variable:

1. Open the example project:

```sh
xed Examples/WhisperAX
```

2. At the top, you will see the app icon and `WhisperAX` written next to it. Click on `WhisperAX` and select `Edit Scheme` at the bottom.

3. Under `Environment Variables`, you will see an entry with `MODEL_NAME` as the name and `$(MODEL_NAME)` as the value.

## Devices

> [!IMPORTANT]
> An active developer account is required to run the tests on physical devices.

Before running tests, all external devices need to be connected and paired to your Mac, as well as registered with your developer account. Ensure the devices are in Developer Mode. If nothing appears after connecting the devices via cable, press `Command + Shift + 2` to open the list of devices and track their progress. 

## Datasets

The datasets for the test suite can be set in a global array called `datasets` in the file [`Tests/WhisperKitTests/RegressionTests.swift`](Tests/WhisperKitTests/RegressionTests.swift). It is prefilled with the datasets that are currently available.

## Models

The models for the test suite can be set in the [`Fastfile`](fastlane/Fastfile). Simply find `BENCHMARK_CONFIGS` and modify the `models` array under the benchmark you want to run.

## Makefile and Fastlane

The tests are run using [Fastlane](fastlane/Fastfile), which is controlled by a [Makefile](Makefile). The Makefile contains the following commands:

### List Connected Devices

Before running the tests it might be a good idea to list the connected devices to resolve any connection issues. Simply run:

```sh
make list-devices
```

The output will be a list with entries that look something like this:

```ruby
{
   :name=>"My Mac", 
   :type=>"Apple M2 Pro", 
   :platform=>"macOS", 
   :os_version=>"15.0.1", 
   :product=>"Mac14,12", 
   :id=>"XXXXXXXX-1234-5678-9012-XXXXXXXXXXXX", 
   :state=>"connected"
}
```

Verify that the devices are connected and the state is `connected`.

### Running Benchmarks

After completing the above steps, you can run the tests. Note that there are two different test configurations: one named `full` and the other named `debug`. To check for potential errors, run the `debug` tests:

```sh
make benchmark-devices DEBUG=true
```

Otherwise run the `full` tests:

```sh
make benchmark-devices
```

Optionally, for both tests, you can specify the list of devices for the tests using the `DEVICES` option:

```sh
make benchmark-devices DEVICES="iPhone 15 Pro Max,My Mac"
```

The `DEVICES` option is a comma-separated list of device names. The device names can be found by running `make list-devices` and using the value for the `:name` key.

### Results

After the tests are run, the generated results can be found under `fastlane/benchmark_data` including the .xcresult file with logs and attachments for each device. There will also be a folder called `fastlane/upload_folder/benchmark_data` that contains only the JSON results in `fastlane/benchmark_data` that can used for further analysis.

We will periodically run these tests on a range of devices and upload the results to the [argmaxinc/whisperkit-evals-dataset](https://huggingface.co/datasets/argmaxinc/whisperkit-evals-dataset), which will propagate to the [WhisperKit Benchmarks](https://huggingface.co/spaces/argmaxinc/whisperkit-benchmarks) space and be available for comparison.


# Troubleshooting


If you encounter issues while running the tests, heres a few things to try:

1. Open the project in Xcode and run the tests directly from there.
   1. To do this, open the example app (from command line type: `xed Examples/WhisperAX`) and run the test named `RegressionTests/testModelPerformanceWithDebugConfig` from the test navigator.
   2. If the tests run successfully, you can rule out any issues with the device or the models.
   3. If they dont run successfully, Xcode will provide more detailed error messages.
2. Try specifying a single device to run the tests on. This can be done by running `make list-devices` and then running the tests with the `DEVICES` option set to the name of the device you want to test on. For example, `make benchmark-devices DEVICES="My Mac"`. This will also enable you to see the logs for that specific device.
3. If you are still encountering issues, please reach out to us on the [Discord](https://discord.gg/G5F5GZGecC) or create an [issue](https://github.com/argmaxinc/argmax-oss-swift/issues) on GitHub.
