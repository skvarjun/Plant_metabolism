from setuptools import setup, find_packages

with open("README.md", "r") as fh:
      description = fh.read()

setup(name='mldice',
      version='0.0.1',
      description='AI Driven Data Analysis on Nanoplastic Influence on Metabolites in Lettuce PLant',
      url='https://github.com/skvarjun/PLant_Metabolism_Nanoplastic',
      author='Arjun S Kulathuvayal',
      author_email='skvarjun@gmail.com',
      license='MIT',
      packages=find_packages(),
      python_requires='>=3',
      install_requires=['numpy==1.24.4', 'scikit-learn==1.3.0', 'seaborn', 'pandas', 'matplotlib'],
      data_files=[('mldice', ['chemical/DNN_model',
                              ])],
      include_package_data=True,
      long_description=description,
      long_description_content_type="text/markdown",
      entry_points={'console_scripts': [
            'pmnp = mldice:MLDiCE'
            ]},
      )