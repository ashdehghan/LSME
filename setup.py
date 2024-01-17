import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
	long_description = fh.read()

setuptools.setup(
	name='LSME',
	version='0.0.1',
	author='Ashkan Dehghan',
	author_email='ash.dehghan@gmail.com',
	description='Local Signature Matrix Embedding',
	long_description=long_description,
	long_description_content_type="text/markdown",
	url='https://github.com/ashdehghan/LSME',
	license='BSD-2',
	packages=['LSME'],
	install_requires=[
	"pandas==2.0.3",
	"tqdm==4.65.0",
	"matplotlib==3.7.2",
	"networkx==2.8.8",
	"numpy==1.25.2"]
)