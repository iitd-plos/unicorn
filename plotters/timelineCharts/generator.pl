#!/usr/bin/perl

use IPC::Open2;

die "Usage: $0 <data file> [<chart title>]\n" if (!($#ARGV == 0 || $#ARGV == 1) || (!-f $ARGV[0]));

my($XPAD) = 0.01;

my(%cancelledSubtasks);
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
    my($minVal) = 0;
    my($maxVal) = 0;
	my(@events);
	my($timelineDefined) = 0;
	my($title);
    my($host);
    
    my($chartTitle) = ($#ARGV == 1) ? $ARGV[1] : "";
	
	open(FH, "$file") || die "Unable to read input file $file\n";

	while(<FH>)
	{
		chomp;
        next if(/^\s*$/);
    
		if(/^PMLIB \[Host [0-9]+\] Event Timeline /)
		{
            my $timelineName = $';
        
            if($timelineDefined == 1)
            {
                storeInGraphCache($title, $host, \@events);
                $#events = -1;
            }
            else
            {
                $timelineDefined = 1;
            }

            $title = removeExtraSpaces($timelineName);
            $title =~ /(.*Device [0-9]+)(.*)/;
            $title = $1."    ";
            $host = removeExtraSpaces($2);
        }
        else
        {
            if(/[0-9.]+\s*$/)
            {
                my $rem = $`;
                my $endTime = removeExtraSpaces($&);
            
                if($rem =~ /[0-9.]+\s*$/)
                {
                    my $startTime = $&;
                    my $eventName = $`;
                    $startTime = removeExtraSpaces($startTime);
                    $eventName = removeExtraSpaces($eventName);
                
                    setMinMaxVals(\$minVal, \$maxVal, $startTime, $endTime);
                
                    my $rec = {
                        EVENT => $eventName,
                        STARTTIME => $startTime,
                        ENDTIME => $endTime
                    };
                
                    if($rec->{EVENT} =~ /([0-9]+)_Cancelled/)
                    {
                        my $subtask = $1;
                        $cancelledSubtasks{$1} = 1;
                    }
                
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

    if($timelineDefined == 1)
    {
        storeInGraphCache($title, $host, \@events);
    }
    
	close FH;
    
    flushGraphCache($minVal, $maxVal, $chartTitle);
}

sub storeInGraphCache
{
	my($title, $host, $eventsRef) = @_;
	my(@events) = @$eventsRef;
    
    my $rec = {
        TITLE => $title,
        HOST => $host,
        EVENTS => [ @events ]
    };
    
    push(@graphCache, $rec);
}

sub flushGraphCache
{
    my($minVal, $maxVal, $chartTitle) = @_;
    
	plotInit($minVal - $XPAD, $maxVal + $XPAD, 1.5 * ($#graphCache + 1) + 1.5, $chartTitle);
    
    @graphCache = sort partialNumericSort @graphCache;
    
    my %hostCounters;
    
    my $i;
    for($i = 0; $i <= $#graphCache; ++$i)
    {
        my $host = $graphCache[$i]->{HOST};

        if($host =~ /\[Host ([0-9]+)\]/)
        {
            $host = $1;
            $hostCounters{$host} = 0 if(!exists $hostCounters{$host});

            ++$hostCounters{$host};
        }

        plotGraph($minVal, $graphCache[$i]->{TITLE}, $graphCache[$i]->{EVENTS}, $i);
    }

    my $startIndex = 0;
    my @hosts = sort {$a <=> $b} keys %hostCounters;
    for($i = 0; $i <= $#hosts; ++$i)
    {
        my $endIndex = $startIndex + $hostCounters{$i};
        my $yMin = 1 + 1.5 * $startIndex;
        my $yMax = (1 + 1.5 * ($endIndex - 1)) + 1;

        writeHostName($yMin, $yMax, $maxVal, $i);
    
        $startIndex = $endIndex;
    }

	plotTerminate();
}

sub plotGraph
{
	my($minXVal, $title, $eventsRef, $graphIndex) = @_;
	my(@events) = @$eventsRef;

    my $yMin = 1 + 1.5 * $graphIndex;
    my $yMax = $yMin + 1;
    
    writeTimelineName($minXVal, $yMin + 0.5, $title);
    
    my $subtask = -1;

    my $i;
    for($i = 0; $i <= $#events; ++$i)
    {
        my $color = 0;
        if($events[$i]->{EVENT} =~ /([0-9]+)_Cancelled/)
        {
            $subtask = $1;
            $color = 1;
        }
        else
        {
            $events[$i]->{EVENT} =~ /([0-9]+)$/;
            $subtask = $1;

            if(exists $cancelledSubtasks{$subtask})
            {
                $color = 2;
            }
        }
    
        plotRect($events[$i]->{STARTTIME}, $yMin, $events[$i]->{ENDTIME}, $yMax, $color, $subtask);
    }
}

sub plotInit
{
    my($minXVal, $maxXVal, $maxYVal, $title) = @_;
    
	print $WH "reset\n";
	print $WH "set terminal svg\n";
	print $WH "set key reverse Left outside\n";
    print $WH "set title \"$title\"\n" if($title !~ /^\s*$/);
    print $WH "set lmargin 10\n";
    print $WH "set rmargin 10\n";
	print $WH "unset ytics\n";
    print $WH "set xlabel \"Time (in secs)\"\n";
	print $WH "set xrange [$minXVal:$maxXVal]\n";
	print $WH "set yrange [0:$maxYVal]\n";
}

sub plotTerminate
{
	print $WH "plot NaN notitle\n";
}

sub plotRect
{
	my($xMin, $yMin, $xMax, $yMax, $color, $subtask) = @_;

    my $xPos = ($xMin + $xMax)/2.0;
    my $yPos = ($yMin + $yMax)/2.0;

    if($color == 1) # cancelled subtask
    {
        print $WH "set obj rect fillcolor rgb \"red\" fillstyle solid 1.0 noborder from $xMin,$yMin to $xMax,$yMax\n";
        print $WH "set label \"$subtask\" at $xPos,$yPos center font \",8\"\n";
    }
    elsif($color == 2) # multi assigned
    {
        print $WH "set obj rect fillcolor rgb \"green\" fillstyle solid 1.0 noborder from $xMin,$yMin to $xMax,$yMax\n";
        print $WH "set label \"$subtask\" at $xPos,$yPos rotate by 90 center font \",8\"\n";
    }
    else
    {
        print $WH "set obj rect fillcolor rgb \"blue\" fillstyle solid 1.0 noborder from $xMin,$yMin to $xMax,$yMax\n";
        print $WH "set label \"$subtask\" at $xPos,$yPos rotate by 90 center font \",8\"\n";
    }
}

sub writeTimelineName
{
    my($minXVal, $yMin, $name) = @_;

    print $WH "set label \"$name\" at $minXVal,$yMin right\n";
}

sub writeHostName
{
    my($minYVal, $maxYVal, $maxXVal, $host) = @_;
    $maxXVal += 0.035;
    my $yVal = ($minYVal + $maxYVal)/2.0;
    
    print $WH "set arrow heads size 0.008,90 from $maxXVal,$minYVal to $maxXVal,$maxYVal\n";
    
    $maxXVal += 0.02;
    print $WH "set label \"Host $host\" at $maxXVal,$yVal rotate by 90 center\n";
}

sub setMinMaxVals
{
    my($minValRef, $maxValRef, $startTime, $endTime) = @_;
    
    $$minValRef = $startTime if($$minValRef == 0 || $startTime < $$minValRef);
    $$maxValRef = $endTime if($endTime > $$maxValRef);
}

sub partialNumericSort
{
    my $arg1 = $a->{TITLE};
    my $arg2 = $b->{TITLE};
    
    $arg1 =~ s/[^0-9]+//;
    $arg2 =~ s/[^0-9]+//;
    
    return $arg1 <=> $arg2;
}

sub removeExtraSpaces
{
	my($str) = @_;

	$str =~ s/^\s+//;
	$str =~ s/\s+$//;
	$str =~ s/\s\s+/ /g;

	return $str;
}

