#!/usr/bin/perl

use IPC::Open2;

die "Usage: $0 <data file>\n" if ($#ARGV != 0 || (!-f $ARGV[0]));

my(@graphCache);

my($RH, $WH, $pid);
$pid = open2($RH, $WH, "gnuplot") || die "Failed to open pipes to gnuplot\n";
plotFile($ARGV[0]);

close $WH;
waitpid($pid, 0);
while(<$RH>)
{
	print $_;
}
close $RH;



sub plotFile
{
	my($file) = @_;
	my($xlabel, $ylabel);
	my(@titles);
	my($labelsDefined) = 0;
	my($titlesDefined) = 0;
	my($lineNum) = 0;
	my($firstLine) = -1;
	my($lastLine) = -1;
	my($title);
	
	open(FH, "$file") || die "Unable to read input file $file\n";

	while(<FH>)
	{
		chomp;
		if(/\|/)
		{
			my(@data) = split(/\|/);
			if($labelsDefined == 0)
			{
				die "Error in x and y labels\n" if ($#data != 1);
				$xlabel = $data[0];
				$ylabel = $data[1];
				$labelsDefined = 1;
			}
			else
			{
				die "Insufficient data\n" if ($#data < 1);
				if($titlesDefined == 0)
				{
					@titles = @data;
					$titlesDefined = 1;
				}
				else
				{
					my($i);
					for($i = 0; $i <= $#data; ++$i)
					{
						$titles[$i] .= " ".$data[$i];
					}
				}
			}
		}
		elsif(!/^=+$/)
		{
			if($labelsDefined == 0)
			{
				$title = removeExtraSpaces($_);
			}
		}
		elsif(/^=+$/)
		{
			if($titlesDefined == 1)
			{
				if($firstLine == -1)
				{
					$firstLine = $lineNum + 1;
				}
				else
				{
					$lastLine = $lineNum - 1;
					storeInGraphCache($title, $xlabel, $ylabel, $file, \@titles, $firstLine, $lastLine);
					$firstLine = $lastLine = -1;
					$labelsDefined = $titlesDefined = 0;
					$#titles = -1;
				}
			}
		}

		++$lineNum;
	}

	close FH;
    
    flushGraphCache();
}

sub storeInGraphCache
{
	my($title, $xlabel, $ylabel, $file, $titlesRef, $firstLine, $lastLine) = @_;
	my(@titles) = @$titlesRef;
    
    my $rec = {
        TITLE => $title,
        XLABEL => $xlabel,
        YLABEL => $ylabel,
        FILE => $file,
        TITLES => [ @titles ],
        FIRSTLINE => $firstLine,
        LASTLINE => $lastLine
    };
    
    push(@graphCache, $rec);
}

sub flushGraphCache
{
	plotInit(1 + $#graphCache, 1);

    my $i;
    for($i = 0; $i <= $#graphCache; ++$i)
    {
        plotGraph($graphCache[$i]->{TITLE}, $graphCache[$i]->{XLABEL}, $graphCache[$i]->{YLABEL}, $graphCache[$i]->{FILE}, \@{$graphCache[$i]->{TITLES}}, $graphCache[$i]->{FIRSTLINE}, $graphCache[$i]->{LASTLINE});
    }

	plotTerminate();
}

sub plotGraph
{
	my($title, $xlabel, $ylabel, $file, $titlesRef, $firstLine, $lastLine) = @_;
	my(@titles) = @$titlesRef;

	$xlabel .= " ".$titles[0];
	$xlabel = removeExtraSpaces($xlabel);
	$ylabel = removeExtraSpaces($ylabel);

	graphInit("$title", "$xlabel", "$ylabel", "$file", $firstLine, $lastLine);

	my($i);
	for($i = 1; $i < $#titles; ++$i)
	{
		plotLine(1, 1+$i, removeExtraSpaces($titles[$i]));
	}

	plotLastLine(1, 1+$i, removeExtraSpaces($titles[$i]));
}

sub plotInit
{
    my($rows, $cols) = @_;
    
	print $WH "reset\n";
	print $WH "set terminal svg\n";
	print $WH "set key reverse Left outside\n";
	print $WH "set grid\n";
	print $WH "set style data linespoints\n";
	print $WH "set multiplot layout $rows, $cols\n";
}

sub plotTerminate
{
	print $WH "unset multiplot\n";
}

sub graphInit
{
	my($title, $xlabel, $ylabel, $dataFile, $firstLine, $lastLine) = @_;

	print $WH "set title \"$title\"\n";
	print $WH "set xlabel \"$xlabel\"\n";
	print $WH "set ylabel \"$ylabel\"\n";

	print $WH "plot \"$dataFile\" every \:\:$firstLine\:\:$lastLine \\\n";
}

sub plotLine
{
	my($xcol, $ycol, $lineName) = @_;

	print $WH "using $xcol:$ycol title \"\", \\\n \"\"";
}

sub plotLastLine
{
	my($xcol, $ycol, $lineName) = @_;

	print $WH "using $xcol:$ycol title \"\" \n";
}

sub removeExtraSpaces
{
	my($str) = @_;

	$str =~ s/^\s+//;
	$str =~ s/\s+$//;
	$str =~ s/\s\s+/ /g;

	return $str;
}

