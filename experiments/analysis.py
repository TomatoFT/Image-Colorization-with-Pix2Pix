import csv
from dataclasses import dataclass

from fid import calculate_fid, load_images


@dataclass
class MethodsEvaluation:
    name: str
    batch_size: int

    def __post_init__(self):
        self.folder = {
            "Pix2Pix": {
                'generated': 'generated/Pix2Pix',
                'original': 'original/Pix2Pix'
            },
            "GFPGAN": {
                'generated': 'experiment_dataset/generated',
                'original': 'experiment_dataset/original'
            },
            "Deoldify": {
                'generated': 'experiment_dataset/deoldify',
                'original': 'experiment_dataset/original'
            }
        }
        self.generated_path = self.folder[self.name]["generated"]
        self.real_path = self.folder[self.name]["original"]

        self.generated_images = load_images(self.generated_path)
        self.real_images = load_images(self.real_path)

    def get_the_fid_score(self) -> float:
        fid_value = calculate_fid(images1=self.generated_images,
                                  images2=self.real_images,
                                  use_multiprocessing=False,
                                  batch_size=self.batch_size
                                  )
        return fid_value


if __name__ == "__main__":
    names = ['Pix2Pix', 'GFPGAN', 'Deoldify']

    with open('report.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Methods', 'FID'])

        for name in names:
            evaluation_instance = MethodsEvaluation(name=name, batch_size=64)
            fid_score = evaluation_instance.get_the_fid_score()
            print(f"FID score of {name} method is {fid_score}")
            writer.writerow([name, fid_score])
