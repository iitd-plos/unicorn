#!/usr/bin/perl

# Developed By Tarun Beri

$|=1;

use File::Basename;
use File::Path;
use Cwd 'abs_path';

${WEB_PAGES} = 1000000;
${MAX_OUTLINKS_PER_WEB_PAGE} = 10;
${WEB_PAGES_PER_FILE} = 1;

${VICINITY_PROBABILITY} = 80; # On scale of 100
${VICINITY} = 5;	# Percentage radius of web considered vicinity (where most links will be concentrated); on scale of 100

${CURRENT_SCRIPT_LOCATION} = abs_path($0);

$scriptBasename = basename($0);
${CURRENT_SCRIPT_LOCATION} =~ s/$scriptBasename//;

${WEB_PATH} = "${CURRENT_SCRIPT_LOCATION}../web_dump";

if($#ARGV != -1)
{
	if($#ARGV != 3)
    {
        die "Usage: WebGenerator.pl <Web pages> <Max outlinks> <Web pages per file> <Web path>";
    }

	${WEB_PAGES} = $ARGV[0];
	${MAX_OUTLINKS_PER_WEB_PAGE} = $ARGV[1];
    ${WEB_PAGES_PER_FILE} = $ARGV[2];
	${WEB_PATH} = $ARGV[3];
    
    die "Potentially wrong web path ${WEB_PATH}" if(${WEB_PATH} !~ /web_dump/);
}

rmtree(${WEB_PATH});
mkpath(${WEB_PATH});

die "Invalid Web Path: ${WEB_PATH}\n" if(!-d "${WEB_PATH}");

${WEB_FILES_PATH} = ${WEB_PATH}."/web";
mkpath(${WEB_FILES_PATH});

die "Invalid Web Path: ${WEB_FILES_PATH}\n" if(!-d "${WEB_FILES_PATH}");

main();

sub main
{
    print "Configuration: $WEB_PAGES web pages; $MAX_OUTLINKS_PER_WEB_PAGE max outlinks per page; $WEB_PAGES_PER_FILE web pages per file\n";
    print "Web Path: $WEB_PATH\n\n";

    generate_web_dump_metadata("$WEB_PATH/web_dump_metadata");

    srand(time());
    
    my($files_to_be_generated) = (${WEB_PAGES} / ${WEB_PAGES_PER_FILE});

    my($i, $file_num);
    for($i = 1, $file_num = 1; $i <= $files_to_be_generated; ++$i, $file_num += ${WEB_PAGES_PER_FILE})
    {
        generate_web_dump_for_pages("$WEB_FILES_PATH/page_${file_num}", ($i - 1) * ${WEB_PAGES_PER_FILE}, ${WEB_PAGES_PER_FILE});
    }
    
    if((${WEB_PAGES} % ${WEB_PAGES_PER_FILE}) != 0)
    {
        my($start_page) = ($i - 1) * ${WEB_PAGES_PER_FILE};
        generate_web_dump_for_pages("$WEB_FILES_PATH/page_${file_num}", $start_page, ${WEB_PAGES} - $start_page);
    }

    print "\nSuccessfully dumped web ...\n\n";
}

# File Format (Binary file containing only integers)
# [Total Number of web pages] [Max outlinks per web page] [Web pages per file]
sub generate_web_dump_metadata
{
	my($metadata_file_path) = $_[0];

	print "Creating web dump metadata at $metadata_file_path ... \n";

    open(FH, ">$metadata_file_path") || die "Failed to open file $metadata_file_path\n";
    binmode(FH);

    print FH pack("L", $WEB_PAGES), pack("L", $MAX_OUTLINKS_PER_WEB_PAGE), pack("L", $WEB_PAGES_PER_FILE);

    close(FH);
}

# File Format (Binary file containing only integers)
# [web page dump] [web page dump] [web page dump] ...
# Web page dump is in the following format
# [Total outlinks from this web page] [Link number of first outlink] [Link number of next outlink] ...
# Total number of dumped links will be max outlinks per web page
# Outlinks number vary from 1 to Total Number of web pages. 0 is a reserved web page number indicating no link.
sub generate_web_dump_for_pages
{
	my($dump_file_path, $start_page, $page_count) = ($_[0], $_[1], $_[2]);

    my($end_page) = $start_page + $page_count;
	print "Creating web dump for pages $start_page to $end_page at $dump_file_path ... \n";

    open(FH, ">$dump_file_path") || die "Failed to open file $dump_file_path\n";
    binmode(FH);

    my($limit) = $MAX_OUTLINKS_PER_WEB_PAGE + 1;

    my($page);
    for($page = ($start_page+1); $page <= $end_page; ++$page)   # 1 based page numbers
    {
        my($count) = int(rand($limit));

        die "Random number out of range\n" if($count > $MAX_OUTLINKS_PER_WEB_PAGE);

        print FH pack("L", $count);

        my($i, $ref_link_num);
        for($i = 0; $i < $count; ++$i)
        {
            my($probability) = int(rand(100));
            if($probability < $VICINITY_PROBABILITY)
            {
                my($vicinity_count) = int(($WEB_PAGES * $VICINITY)/100);

                my($vicinity_num) = int(rand($vicinity_count));

                my($forward) = int(rand(10));	# With equal probability the link may be to next or prev that many pages
                if($forward < 5)
                {
                    $ref_link_num = $page + $vicinity_num;

                    if($ref_link_num > $WEB_PAGES)
                    {
                        $ref_link_num = $page - $vicinity_num;
                    }
                }
                else
                {
                    $ref_link_num = $page - $vicinity_num;

                    if($ref_link_num <= 0)
                    {
                        $ref_link_num = $page + $vicinity_num;
                    }
                }
            }
            else
            {
                $ref_link_num = 1 + int(rand($WEB_PAGES));     # 1 based indexing of web pages
            }

            print FH pack("L", $ref_link_num);
        }

        for($i = $count; $i < $MAX_OUTLINKS_PER_WEB_PAGE; ++$i)
        {
            print FH pack("L", 0);
        }
    }

    close(FH);
}
