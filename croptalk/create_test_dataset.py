from langsmith import Client

from dotenv import load_dotenv
load_dotenv('secrets/.env.secret')
load_dotenv('secrets/.env.shared')


def create_dataset_from_csv(csv_file, name, description, input_keys, output_keys):

    client = Client()

    # TODO: add check if dataset exists

    dataset = client.upload_csv(
        csv_file=csv_file,
        input_keys=input_keys,
        output_keys=output_keys,
        name=name,
        description=description,
        data_type="kv"
    )

    return dataset


if __name__ == "__main__":

    filepath = 'data/questions.csv'
    name = "Default Dataset"
    description = "Default dataset to test CropTalk Knowledge"
    input_keys = ['Question', 'Context']  # a-question and user-context
    output_keys = ['Answer', 'QASource']  # answer and source

    ds = create_dataset_from_csv(
        filepath, name, description, input_keys, output_keys)
    print(ds)
