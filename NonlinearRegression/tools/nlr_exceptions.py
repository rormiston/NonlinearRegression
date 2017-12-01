import os
import textwrap


class TemplateNotFound(Exception):
    """
    TemplateNotFound is used for catching exceptions
    when jinja templates are not found
    """
    def __init__(self, template):

        self.template = template

        message = ("The jinja template '{}' could not be found. "
                   "Make sure that you're sourced and the template "
                   "exists").format(template)

        wrapper = textwrap.TextWrapper(width=100)
        message = wrapper.fill(text=message)

        # Call the base class constructor with the parameters it needs
        super(TemplateNotFound, self).__init__(message)


class FileNotFound(Exception):
    """
    FileNotFound is used for catching exceptions when the file that is
    attempting to be opened does not exist.
    """
    def __init__(self, FILE):

        self.FILE = FILE

        message = ("{} does not exist. Most likely, the given path is "
                   "incorrect. Try specifying it with the '-i' flag")

        wrapper = textwrap.TextWrapper(width=100)
        message = wrapper.fill(text=message)

        # Call the base class constructor with the parameters it needs
        super(FileNotFound, self).__init__(message.format(FILE))


class DataNotFound(Exception):
    """
    FileNotFound is used for catching exceptions when the file that is
    attempting to be opened does not exist.
    """
    def __init__(self, datafile):

        self.datafile = datafile

        message = ("{} could not be found. Make sure that the data "
                   "is stored in timedelay/Data/")

        wrapper = textwrap.TextWrapper(width=100)
        message = wrapper.fill(text=message)

        # Call the base class constructor with the parameters it needs
        super(DataNotFound, self).__init__(message.format(datafile))


def loadTemplate(env, html_template):
    """
    loadTemplate tries to load the given html template to be used
    when building the webpages
    """
    try:
        template = env.get_template(html_template)
        return template
    except:
        raise TemplateNotFound(html_template)


def checkFileExists(ini_file):
    """
    checkFileExists throws an exception if the file does not exist.
    This prevents misleading errors when trying to read from secitons
    in config files that do not exist.
    """
    if not os.path.isfile(ini_file):
        raise FileNotFound(ini_file)
    else:
        pass
