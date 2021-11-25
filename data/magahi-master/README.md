# Magahi

This repository contains all the data, tools, applications and publications related to Magahi, an Indo-Aryan language spoken by approx. 11 million speakers largely in the Eastern Indian state of Bihar. Currently 'data' contains both Part-of-Speech annotated (in the directory called 'annotated') as well as raw data (in the directory called 'raw'). The annotated data contains gold data (i.e. manually annotated data) in the directory 'gold-pos' as well as automatically annotated data using the Magahi part-of-speech tagger (available at http://www.kmiagra.org/magahi-pos) in the directory called 'auto-pos'. The same data is also available inside the 'conll-datasets' directory. Additionally an online search engine is also available at http://www.kmiagra.org/msearchit for searching through the raw corpus.

The tools directory contains the following -
Magahi Morph Analyser (magahi-morph) - a rule-based analyser. 
Magahi BIS POS Tagger (magahi-pos) - a maximum entropy based pos-tagger.
Corpus Search Tool (msearchit) - searches through the specified corpus; comes with a support for regex

All of these require Apache Tomcat to run. Just copy the directory inside the 'webapps' of Apache Tomcat Server and it should work out-of-the-box. If it doesnt then try compiling the Java classes again and also check the permissions for the libraries inside the 'lib' folder. These have been tested for Apache Tomcat 8 (may not work for higher versions) on Ubuntu 18.04 (may not work for other OS)

In addition to this, it also contain 'ud-pos-tagger' - POS tagger for annotating with Universal Dependencies POS tags. It requires Python3 and scikit-learn to work. The input must be in CONLL format.

The 'utils' directory contains some Python scripts for mapping BIS tagged data to UD and some other file format converters.

All these resources, tools and data have been developed as part of the research done by Mr. Deepak Alok, Dr. Bornini Lahiri and Dr. Ritesh Kumar under an unfunded project called 'Automatic Language Understanding and Processing Resources for Magahi (ALUP – M)'.

These are being made available for further research and expansion at no cost. However, if you are interested in making a commercial usage of these resources, please contact the creators of this project.

More tools and data is expected to be added and these ones modified in the future.


For any queries, please send an email to riteshkrjnu[at]gmail[dot]com
