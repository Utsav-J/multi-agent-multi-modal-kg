from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions,
)
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

pipeline_options = PdfPipelineOptions()
pipeline_options.do_picture_description = True
source = "attention_is_all_you_need.pdf"
converter = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
})
pipeline_options = ThreadedPdfPipelineOptions(
    accelerator_options=AcceleratorOptions(
        device=AcceleratorDevice.CUDA,
    ),
    ocr_batch_size=4,
    layout_batch_size=64,
    table_batch_size=4,
)

result = converter.convert(source)

full_text = result.document.export_to_markdown()
with open("output.md", "w", encoding='utf-8') as f:
        f.write(full_text)