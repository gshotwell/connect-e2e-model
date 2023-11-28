# End-to-end model management 

This is a demo repository showing an end-to-end model management workflow with Posit Connect. 

![]("diagram.png")

There are four main parts to this system. 

## The model API

The [model api](https://colorado.posit.co/rsc/electronics-classifier/) is at the center of this system, and can be found in the `api` subdirectory. 
It is written using FastAPI and each endpoint is authorized against the Connect user id. 
This lets you distribute an API to a large group of people, while ensuring that only authorized individuals are able to interact with certain enpoints. 

There are a couple of things that you should be aware of when using this pattern:

1) This authroization pattern is meant to protect against accidental interaction with the wrong endpoint, not malicious attack. You should use this pattern in addition to Connect's security features to ensure that the top level API domain can only be accessed by the right users. 

2) For simplicity the API stores data and models in the content bundle, but you probably want to use external storage locations like S3 or a database table. 
Using extenral storage will mean that your data and files will persist regardless of the API deployment, and is cheaper and more robust than storing data on Connect.

## The Python package

The python package wraps the API and provides user-friendly error messages and type annotations. 
You could deploy this package to an internal repository like Posit Package Manager, or just install it from github. 

## Shiny annotator

[The annotator](https://colorado.posit.co/rsc/electronics-annotator/) is designed to let people quickly annotate training data. 
It uses the API endpoints to retrieve, score, and update text. 