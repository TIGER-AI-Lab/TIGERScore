### Human Evaluation Data Collection: AMT Integration Scripts

# Overview

Scripts for setting up and running human evaluation experiments on Amazon Mechanical Turk conducted for the [WebNLG+ 2020 Shared Task](https://webnlg-challenge.loria.fr/challenge_2020/).

# General Information

The backbone of the data collection pipeline is based on the frameworks developed for [dialogue data collection](https://clp-research.github.io/slurk/slurk_amt.html) and [simple-amt framework](https://github.com/jcjohnson/simple-amt).
We use AMT API to run the experiments, for more information on specific operations please check official [AMT API documentation](https://docs.aws.amazon.com/AWSMechTurk/latest/AWSMturkAPI/Welcome.html).

# Preparing data collection

0. Install the requirements from `requirements.txt` by running `pip install -r requirements.txt`
Note: do not forget to clone the submodule as well (https://gitlab.com/webnlg/corpus-reader) for corpus reading. You can do it by running `git submodule update --init --recursive`.

1. In our experiments, we display triplets as images to the participants. We first generate the triplet tables in XML format and then create images from them. For more details, please check `scripts/prepare_tables.ipynb` and `scripts/compile_tables.ipynb`. The images should be stored in the publicly accessible GitHub repository in order to display them on AMT (you should simply change their storage location to achieve this).

2. Edit the configuration file. We provide a template configuration file `config_example.json`. Configuration file requires the following arguments:

| Argument        | Description           |
| --------------- |-------------|
| `qual_triplet_weblink` |   link to publicly available .jpg images of triplets for qualification task |
| `qual_triplet_localpath` | local path to the .jpg images of qualification triplets |
| `qual_description_localpath` |  local path to qualification triplet descriptions |
| `orig_triplet_weblink` |   link to publicly available .jpg images of actual (original) data triplets |
| `orig_triplet_localpath` | local path to the .jpg images of original triplets |
| `orig_description_localpath` |  local path to the original triplet descriptions |
| `aws_access_id` | part of the AWS security credentials |
| `aws_secret_key` |  part of the AWS security credentials |

Arguments with `qual` in front of them are used when publishing qualification tasks (add qualification task information here)

AWS security credentials are required to be able to login to your AMT account programmatically.
For more information on setting up AWS security credentials, check [Setting Up Accounts and Tools](https://docs.aws.amazon.com/AWSMechTurk/latest/AWSMechanicalTurkGettingStartedGuide/SetUp.html).

---

**To run data collection**, you are required to run bash scripts with either `publish` or `review` in their names. `qual` in the name of the bash script indicates that they are used for qualification tasks.
`publish` files are used to release tasks on AMT, while `review` files extract intermediate results and assign qualifications to the workers.
Note: you need to run `publish` and `review` simultaneously. Thus, we recommend to use two *screen* sessions (https://www.gnu.org/software/screen/manual/screen.html) and run them in the background of your system.

---
**Publishing tasks**
`publish.py` takes the following arguments:

| Argument        | Description           |
| --------------- |-------------|
| `hit_properties_file` | HIT metadata (title, description, etc.)  |
| `html_template` | html layout of the HIT |
| `input_json_file` |  the set of some HIT data (incl. text, image link, submission id, triplet size, etc.), used to differentiate between different triplet-text pairs and submissions  |
| `hit_ids_file` | .txt file that is constantly updated with newly published HIT ids |
| `prod` | when specified, publish tasks in the production (actual AMT) environment; otherwise, publish tasks in Sandbox (testing environment)  |

The file for the `input_json_file` argument is created by running `generate.py`.
The example of the generated file can be found in `./data/en/inputs/hit_inputs.json`.
