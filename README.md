# Truncated Recursive Autoassociative Chunk eXtractor

TRACX is a new model of sequence learning. It is a connectionist autoassociator model which fits a wide range of phenomena from the infant statistical learning and adult implicit learning literature. TRACX outperforms PARSER (Perruchet & Vintner, 1998) and the simple recurrent network (SRN, Cleeremans & McClelland, 1991) in matching human sequence segmentation on existing data. This simulator is written entirely in Javascript and runs locally in your browser. A short summary of the algorithm can be found here and more details of the model can be found in:

    French, R. M., Addyman, C., & Mareschal, D. (2011). TRACX: A recognition-based connectionist framework for sequence segmentation and chunk extraction *Psychological Review*, 118(4), 614–636. doi:10.1037/a0025255 

This is a small help file explaining how to get started using the TRACX online simulator code.


Requirements:
=============

TRACX utilises several Javascript libraries. Pre-compiled versions are included in the /lib/ folder

JQuery 1.4.2 or greater:
http://jquery.com/

JQPlot 1.0.0b2 or greater:
http://www.jqplot.com/

Sylvester 0.1.3 or greater
http://sylvester.jcoglan.com/

David Bau's Random Seed function
http://davidbau.com/archives/2010/01/30/random_seeds_coded_hints_and_quintillions.html

Optional:
=========

JQTip 1.0.0-rc3 or greater: 
http://craigsworks.com/projects/qtip/
                      



1. Simulator Only
==================
If you just wish to use the simulator, you can try it online at:
http://leadserv.u-bourgogne.fr/~tracx/


2. Local debugging
==================

If you just wish to use the simulator _and_ step through the code. You can use the current live code from:
http://leadserv.u-bourgogne.fr/~tracx/

Together with an in browser debug engine. We recommend Firebug:
https://getfirebug.com/

We have a beginners guide to debugging available at:
http://leadserv.u-bourgogne.fr/~tracx/debughelp.html


3. Adapting and modifying the TRACX code yourself
=================================================

This code is open source and freely available for you to use in your own projects. Just download the code and away you go. We recommend the Aptana Development environment.

Aptana :
http://aptana.org/

Within Aptana you can get started quickly by choosing the File/Import.. menu. Project type is __git__ and the repository URI is:

99% of the code is client-side javascript so will work locally. If you want to get the php working (to load up default datasets) we recommend

XAMMP:
http://www.apachefriends.org/en/xampp.html

That's it.

Any questions, bugs or comments, please post them on github.com or email them to c.addyman@bbk.ac.uk







