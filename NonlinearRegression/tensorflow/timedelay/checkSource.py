"""
This script checks if you are sourced and attempts to do so
if you are not. Relative imports will not work if you are not
sourced so this file needs to stay here.
"""
import subprocess
import os
import textwrap


class SourceError(Exception):
    """
    SourceError is used for catching exceptions when a script
    that requires the noise_cancellation virtual environment
    has attempted run outside of the VE.
    """
    def __init__(self):

        message = ("You need to source the virtual environment "
                   "before continuing. Run: source path-to-VE/bin/activate")

        wrapper = textwrap.TextWrapper(width=100)
        message = wrapper.fill(text=message)

        # Call the base class constructor with the parameters it needs
        super(SourceError, self).__init__(message)


def checkSource():
    """
    checkSource makes sure that the virtual environment is sourced before
    running the pipeline. If it is not, it will attempt to source it for
    you. Assuming a default install, this should work. Otherwise, the
    SourceError exception will be thrown.
    """
    out = subprocess.Popen('echo $VIRTUAL_ENV',
                           stdout = subprocess.PIPE,
                           shell  = True)

    out = out.stdout.read().strip('\n')

    if len(out) == 0:
        try:
            HOME     = os.getenv('HOME')
            activate = HOME + '/noise_cancellation/bin/activate_this.py'
            execfile(activate, dict(__file__ = activate))

            warning  = ("WARNING: You were not sourced before running this "
                        "script and have been automatically sourced. The "
                        "virtual environment will deactivate after the "
                        "execution of this script ends.\n")

            wrapper = textwrap.TextWrapper(width=100)
            warning = wrapper.fill(text=warning)
            print(warning)

        except:
            raise SourceError()
    else:
        pass
