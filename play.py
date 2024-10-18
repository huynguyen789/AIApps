import base64
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from subprocess import PIPE, run
from zipfile import ZipFile

import streamlit as st

st.set_page_config(page_title="Office to PDF Converter",
                    page_icon='📄',
                    layout='wide',
                    initial_sidebar_state='expanded')

# apply custom css if needed
with open(Path('utils/style.css')) as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)


@st.cache_resource(ttl=60*60*24)
def cleanup_tempdir() -> None:
    '''Cleanup temp dir for all user sessions.
    Filters the temp dir for uuid4 subdirs.
    Deletes them if they exist and are older than 1 day.
    '''
    deleteTime = datetime.now() - timedelta(days=1)
    # compile regex for uuid4
    uuid4_regex = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    uuid4_regex = re.compile(uuid4_regex)
    tempfiledir = Path(tempfile.gettempdir())
    if tempfiledir.exists():
        subdirs = [x for x in tempfiledir.iterdir() if x.is_dir()]
        subdirs_match = [x for x in subdirs if uuid4_regex.match(x.name)]
        for subdir in subdirs_match:
            itemTime = datetime.fromtimestamp(subdir.stat().st_mtime)
            if itemTime < deleteTime:
                shutil.rmtree(subdir)


# TODO: this function is unused yet
def cleanup_session_tempdir() -> None:
    '''Cleanup temp dir for user session.
    Deletes the whole session temp dir if it exists.
    '''
    if 'tempfiledir' in st.session_state:
        tempfiledir = st.session_state['tempfiledir']
        if tempfiledir.exists():
            shutil.rmtree(tempfiledir)


# TODO: this function is unused yet
def get_all_subdirs_in_tempdir() -> list:
    '''Get all subdirs in temp dir
    returns: list of subdirs
    '''
    tempfiledir = Path(tempfile.gettempdir())
    subdirs = [x for x in tempfiledir.iterdir() if x.is_dir()]
    return subdirs


@st.cache_data(show_spinner=False)
def make_tempdir() -> Path:
    '''Make temp dir for each user session and return path to it
    returns: Path to temp dir
    '''
    if 'tempfiledir' not in st.session_state:
        tempfiledir = Path(tempfile.gettempdir())
        tempfiledir = tempfiledir.joinpath(f"{uuid.uuid4()}")   # make unique subdir
        tempfiledir.mkdir(parents=True, exist_ok=True)  # make dir if not exists
        st.session_state['tempfiledir'] = tempfiledir
    return st.session_state['tempfiledir']


def store_file_in_tempdir(tmpdirname: Path, uploaded_file: BytesIO) -> Path:
    '''Store file in temp dir and return path to it
    params: tmpdirname: Path to temp dir
            uploaded_file: BytesIO object
    returns: Path to stored file
    '''
    # store file in temp dir
    tmpfile = tmpdirname.joinpath(uploaded_file.name)
    with open(tmpfile, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return tmpfile


def convert_doc_to_pdf_native(doc_file: Path, output_dir: Path=Path("."), timeout: int=60):
    """Converts a doc file to pdf using libreoffice without msoffice2pdf.
    Calls libroeoffice (soffice) directly in headless mode.
    params: doc_file: Path to doc file
            output_dir: Path to output dir
            timeout: timeout for subprocess in seconds
    returns: (output, exception)
            output: Path to converted file
            exception: Exception if conversion failed
    """
    exception = None
    output = None
    try:
        process = run(['soffice', '--headless', '--convert-to',
            'pdf:writer_pdf_Export', '--outdir', output_dir.resolve(), doc_file.resolve()],
            stdout=PIPE, stderr=PIPE,
            timeout=timeout, check=True)
        stdout = process.stdout.decode("utf-8")
        re_filename = re.search('-> (.*?) using filter', stdout)
        output = Path(re_filename[1]).resolve()
    except Exception as e:
        exception = e
    return (output, exception)


@st.cache_data(show_spinner=False)
def get_base64_encoded_bytes(file_bytes) -> str:
    base64_encoded = base64.b64encode(file_bytes).decode('utf-8')
    return base64_encoded


@st.cache_data(show_spinner=False)
def show_pdf_base64(base64_pdf):
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="1000px" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_versions() -> str:
    result = run(["soffice", "--version"], capture_output=True, text=True)
    libreoffice_version = result.stdout.strip()
    versions = f'''
    - `Streamlit {st.__version__}`
    - `{libreoffice_version}`
    '''
    return versions


def get_all_files_in_tempdir(tempfiledir: Path) -> list:
    files = [x for x in tempfiledir.iterdir() if x.is_file()]
    files = sorted(files, key=lambda f: f.stat().st_mtime)
    return files


def get_pdf_files_in_tempdir(tempfiledir: Path) -> list:
    files = [x for x in tempfiledir.iterdir() if x.is_file() and x.suffix == '.pdf']
    files = sorted(files, key=lambda f: f.stat().st_mtime)
    return files


def get_zip_files_in_tempdir(tempfiledir: Path) -> list:
    files = [x for x in tempfiledir.iterdir() if x.is_file() and x.suffix == '.zip']
    files = sorted(files, key=lambda f: f.stat().st_mtime)
    return files


def make_zipfile_from_filelist(filelist: list, output_dir: Path=Path("."), zipname: str="Converted.zip") -> Path:
    """Make zipfile from list of files
    params: filelist: list of files
            output_dir: Path to output dir
            zipname: name of zipfile
    returns: Path to zipfile
    """
    zip_path = output_dir.joinpath(zipname)
    # check if filelist is empty and don't create zipfile resp. delete it
    if not filelist:
        zip_path.unlink(missing_ok=True)
        return None
    else:
        with ZipFile(zip_path, 'w') as zipObj:
            for file in filelist:
                zipObj.write(file, file.name)
        return zip_path


# TODO: this function is unused yet
def delete_all_pdf_files_in_tempdir(tempfiledir: Path):
    for file in get_pdf_files_in_tempdir(tempfiledir):
        file.unlink()


# TODO: this function is unused yet
def delete_all_zip_files_in_tempdir(tempfiledir: Path):
    for file in get_zip_files_in_tempdir(tempfiledir):
        file.unlink()


def delete_all_files_in_tempdir(tempfiledir: Path):
    for file in get_all_files_in_tempdir(tempfiledir):
        file.unlink()


def delete_files_from_tempdir_with_same_stem(tempfiledir: Path, file_path: Path):
    file_stem = file_path.stem
    for file in get_all_files_in_tempdir(tempfiledir):
        if file.stem == file_stem:
            file.unlink()


def get_bytes_from_file(file_path: Path) -> bytes:
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    return file_bytes


def check_if_file_with_same_name_and_hash_exists(tempfiledir: Path, file_name: str, hashval: int) -> bool:
    """Check if file with same name and hash already exists in tempdir
    params: tempfiledir: Path to file
            file_name: name of file
            hashval: hash of file
    returns: True if file with same name and hash already exists in tempdir
    """
    file_path = tempfiledir.joinpath(file_name)
    if file_path.exists():
        file_hash = hash((file_path.name, file_path.stat().st_size))
        if file_hash == hashval:
            return True
    return False


def show_sidebar():
    with st.sidebar:
        # st.image('pdf.png', width=50)
        st.header('About')
        st.markdown('''This app can convert **Microsoft Word** or **LibreOffice Writer** Documents to PDF.''')
        st.markdown('''Supported input file formats are:''')
        st.markdown('''`docx`, `doc`, `odt`, `rtf`''')
        st.markdown('''---''')
        st.subheader('Disclaimer')
        st.markdown('''The conversion is done using a headless version of **LibreOffice**.
            As we all know, there are some compatibility issues between LibreOffice and Microsoft Office.
            Therefore, the conversion might not be perfect and may fail in some cases.
            Especially if you convert a Microsoft Office document and/or the document is more complex.''')
        st.markdown('''---''')
        st.subheader('Versions')
        st.markdown(get_versions(), unsafe_allow_html=True)
        st.markdown('''---''')
        st.subheader('GitHub')
        st.markdown('''<https://github.com/Franky1/Streamlit-docx-converter>''')


if __name__ == "__main__":
    cleanup_tempdir()  # cleanup temp dir from previous user sessions
    tmpdirname = make_tempdir()  # make temp dir for each user session
    show_sidebar()
    headercol1, headercol2 = st.columns([8,1], gap='large')
    with headercol1:
        st.title('Office to PDF Converter 📄')
    with headercol2:
        st.image('resources/pdf.png', width=60)
    st.markdown('''---''')
    # add streamlit 2 column layout
    col1, col2 = st.columns([6,8], gap='large')
    pdf_file = None
    pdf_bytes = None
    with col1:
        st.subheader('Upload MS Office or LibreOffice file(s)')
        with st.form("convert", clear_on_submit=True):
            uploaded_files = st.file_uploader(label='Upload Microsoft Word or LibreOffice Writer file(s)',
                    type=['doc', 'docx', 'odt', 'rtf'],
                    accept_multiple_files=True,
                    key='file_uploader')
            submitted = st.form_submit_button("Convert uploaded file(s) to PDF")
        if submitted and uploaded_files is not None:
            for uploaded_file in uploaded_files:
                if uploaded_file is not None:
                    uploaded_file_hash = hash((uploaded_file.name, uploaded_file.size))
                    if check_if_file_with_same_name_and_hash_exists(tempfiledir=tmpdirname, file_name=uploaded_file.name, hashval=uploaded_file_hash) is False:
                        # store file in temp dir
                        tmpfile = store_file_in_tempdir(tmpdirname, uploaded_file)
                        # convert file to pdf
                        with st.spinner('Converting file...'):
                            pdf_file, exception = convert_doc_to_pdf_native(doc_file=tmpfile, output_dir=tmpdirname)
                        if exception is not None:
                            st.exception('Exception occured during conversion.')
                            st.exception(exception)
                            st.stop()
                        elif pdf_file is None:
                            st.error('Conversion failed. No PDF file was created.')
                            st.stop()
                        elif pdf_file.exists():
                            st.success(f"Conversion successful: {pdf_file.name}")
                            pdf_bytes = get_bytes_from_file(pdf_file)

    with col2:
        st.subheader('Preview/Download/Delete converted PDF file(s)')
        pdf_files_in_temp = get_pdf_files_in_tempdir(tmpdirname)
        # make subcolums for download or deleting single pdf files
        if len(pdf_files_in_temp) > 0:
            file_bytes = list()
            for index, file in enumerate(pdf_files_in_temp):
                file_bytes.append(get_bytes_from_file(file))
                subcol1, subcol2, subcol3, subcol4 = st.columns([12,3,3,3])
                with subcol1:
                    st.info(file.name)
                with subcol2:
                    if st.button('Preview', key=f'preview_button_{index}'):
                        pdf_file = file
                        pdf_bytes = file_bytes[index]
                with subcol3:
                    st.download_button(label='Download',
                                data=file_bytes[index],
                                file_name=file.name,
                                mime='application/octet-stream',
                                key=f'download_button_{index}')
                with subcol4:
                    if st.button('Delete', key=f'delete_button_{index}'):
                        delete_files_from_tempdir_with_same_stem(tmpdirname, file)
                        st.experimental_rerun()
            # download button for all pdf files as zip
            st.markdown('''<br>''', unsafe_allow_html=True)
            zip_path = make_zipfile_from_filelist(pdf_files_in_temp, tmpdirname)
            zip_bytes = get_bytes_from_file(zip_path)
            st.download_button(label='Download all PDF files as single ZIP file',
                        data=zip_bytes,
                        file_name=zip_path.name,
                        mime='application/octet-stream')
            if st.button('Delete all files from Temporary folder', key='delete_all_button'):
                delete_all_files_in_tempdir(tmpdirname)
                st.experimental_rerun()
        else:
            st.warning('No PDF files available for download.')

    if pdf_bytes is not None:
        st.markdown('''---''')
        st.subheader(f'Preview of converted PDF file "{pdf_file.name}"')
        pdf_bytes_base64 = get_base64_encoded_bytes(pdf_bytes)
        # show pdf in iframe already base64 encoded
        show_pdf_base64(pdf_bytes_base64)