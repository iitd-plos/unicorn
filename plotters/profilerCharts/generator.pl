#!/usr/bin/perl

use IPC::Open2;

die "Usage: $0 <data file> [<chart title>]\n" if (!($#ARGV == 0 || $#ARGV == 1) || (!-f $ARGV[0]));

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
    my($maxVal) = 0;
	my(@events);
	my($hostDefined) = 0;
	my($title);
    
    my($chartTitle) = ($#ARGV == 1) ? $ARGV[1] : "";
	
	open(FH, "$file") || die "Unable to read input file $file\n";

	while(<FH>)
	{
		chomp;
        next if(/^\s*$/);

		if(/^Task Profiler (\[Host [0-9]+\]) /)
		{
            my $tmp = $1;
            $tmp =~ s/[\[\]]//g;
        
            if($hostDefined == 1)
            {
                storeInGraphCache($title, \@events);
                $#events = -1;
            }
            else
            {
                $hostDefined = 1;
            }

            $title = $tmp;
        }
        else
        {
            if(/^([A-Z_]+) => Accumulated Time: (.+)s; Actual Time = (.+)s; Overlapped Time = (.+)s/)
            {
                if($hostDefined == 1)
                {
                    my $eventName = $1;
                    my $accumulatedTime = $2;
                    my $actualTime = $3;
                    my $overlappedTime = $4;
                
                    setMaxVal(\$maxVal, $maxVal, $accumulatedTime, $actualTime, $overlappedTime);
                
                    my $rec = {
                        EVENT => $eventName,
                        ACCUMULATED_TIME => $accumulatedTime,
                        ACTUAL_TIME => $actualTime,
                        OVERLAPPED_TIME => $overlappedTime
                    };

                    push(@events, $rec);
                }
                else
                {
                    die "Error in data\n";
                }
            }
            else
            {
				die "Error in data\n";
            }
		}
	}

    if($hostDefined == 1)
    {
        storeInGraphCache($title, \@events);
    }
    
	close FH;
    
    flushGraphCache($maxVal, $chartTitle);
}

sub storeInGraphCache
{
	my($title, $eventsRef) = @_;
	my(@events) = @$eventsRef;
    
    my $rec = {
        TITLE => $title,
        EVENTS => [ @events ]
    };
    
    push(@graphCache, $rec);
}

sub getCount
{
    my($eventsRef) = @_;
    my(@events) = @$eventsRef;
    
    return ($#events + 1);
}

sub flushGraphCache
{
    my($maxVal, $chartTitle) = @_;

    my $firstEventsRef = $graphCache[0]->{EVENTS};
    my $eventsCount = $#$firstEventsRef + 1;

	plotInit($eventsCount* ($#graphCache + 1) + 2 * $eventsCount, $maxVal * 1.15, $chartTitle);
    
    @graphCache = sort partialNumericSort @graphCache;
    
    my $i;
    for($i = 0; $i <= $#graphCache; ++$i)
    {
        plotGraph($graphCache[$i]->{TITLE}, $graphCache[$i]->{EVENTS}, $i, $maxVal);
    }

    my $xCurr = 1;
    my $xSpan = $#graphCache + 1;
    my $xStep = $xSpan + 2;
    for($i = 0; $i < $eventsCount; ++$i)
    {
        writeEventName($xCurr + $xSpan, $firstEventsRef->[$i]->{EVENT});
        $xCurr += $xStep;
    }

	plotTerminate();
}

sub plotGraph
{
	my($title, $eventsRef, $graphIndex, $yMax) = @_;
	my(@events) = @$eventsRef;

    my $xSpan = 1;
    my $xCurr = 1 + $xSpan * $graphIndex;
    my $xStep = ($#graphCache + 1) + 2;
    
    my $color = $graphIndex;
    
    my $i;
    for($i = 0; $i <= $#events; ++$i)
    {
        plotBar($events[$i]->{ACCUMULATED_TIME}, $events[$i]->{ACTUAL_TIME}, $color, $xCurr, $xCurr + $xSpan, $title, $yMax);
        $xCurr += $xStep;
    }
}

sub plotInit
{
    my($maxXVal, $maxYVal, $title) = @_;

	print $WH "reset\n";
	print $WH "set terminal svg\n";
    print $WH "set style histogram\n";
	print $WH "set key reverse Left outside\n";
    print $WH "set title \"$title\"\n" if($title !~ /^\s*$/);
    print $WH "set lmargin 10\n";
    print $WH "set rmargin 10\n";
    print $WH "set bmargin 5\n";
    print WH "set tics scale 0.0\n";
	print $WH "unset xtics\n";
    print $WH "set ytics nomirror\n";
    print $WH "set grid y lt 0\n";
    print $WH "set ylabel \"Time (in secs)\"\n";
	print $WH "set xrange [0:$maxXVal]\n";
	print $WH "set yrange [0:$maxYVal]\n";
}

sub plotTerminate
{
	print $WH "plot NaN notitle\n";
}

sub plotBar
{
	my($accumulatedTime, $actualTime, $color, $xMin, $xMax, $label, $yMax) = @_;
    my(@darkColors) = ("#d2691e", "#ff7f50", "#8a2be2", "#a52a2a", "#0000ff", "#6495ed", "#dc143c", "#00008b", "#008b8b", "#b8860b", "#006400", "#8b008b", "#556b2f", "#9932cc", "#8b0000", "#483d8b", "#2f4f4f");
    my(@lightColors) = ("#f0f8ff", "#faebd7", "#00ffff", "#7fffd4", "#f0ffff", "#f5f5dc", "#ffe4c4", "#ffebcd", "#deb887", "#5f9ea0", "#7fff00", "#fff8dc", "#00ffff", "#a9a9a9", "#bdb76b", "#ff8c00", "#e9967a");
    my $color1 = $darkColors[$color % ($#darkColors + 1)];
    my $color2 = $lightColors[$color % ($#lightColors + 1)];
    print $WH "set obj rect fillcolor rgb \"$color1\" fillstyle solid 1.0 noborder from $xMin,0 to $xMax,$accumulatedTime\n";
    
    $xMin += 0.25;
    $xMax -= 0.25;
    print $WH "set obj rect fillcolor rgb \"$color2\" fillstyle solid 1.0 noborder from $xMin,0 to $xMax,$actualTime\n";
    
    my $yPos = $accumulatedTime;
    if($yPos > 0)
    {
        my $xPos = ($xMin + $xMax)/2.0;
        print $WH "set label \"  $label\" at $xPos,$yPos rotate by 90 font \",7\"\n";
    }
}

sub writeEventName
{
    my($xVal, $name) = @_;

    print $WH "set label \"$name     \" at $xVal,0 rotate by 45 right font \",7\"\n";
}

sub setMaxVal
{
    my($maxValRef, $currMaxVal, $accumulatedTime, $actualTime, $overlappedTime) = @_;
    
    $$maxValRef = getMax(getMax(getMax($currMaxVal, $accumulatedTime), $actualTime), $overlappedTime);
}

sub getMax
{
    my($val1, $val2) = @_;
    
    return $val1 if($val1 > $val2);
    return $val2;
}

sub partialNumericSort
{
    my $arg1 = $a->{TITLE};
    my $arg2 = $b->{TITLE};
    
    $arg1 =~ s/[^0-9]+//;
    $arg2 =~ s/[^0-9]+//;
    
    return $arg1 <=> $arg2;
}
