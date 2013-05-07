#!/usr/bin/perl

use Cwd 'abs_path';
my($script_path) = abs_path($0);

$script_path =~ /(.*)\/.*\/.*$/;
my($pm_base_path) = $1;

my($linux) = `uname -a | grep Linux`;
chomp($linux);


my($runLevel, $parallelTaskMode, $schedulingModel, $samples, $minProcs, $maxProcs, $hostsFile, $exec_varyings, $varying1_min, $varying1_max, $varying1_step, $varying2_min, $varying2_max, $varying2_step);

main();

sub main
{
    my($testSuite) = getTestSuite();
    
    my($exec_path) = "$pm_base_path/testSuite/$testSuite/build/linux/release/$testSuite.exe";
    die "Invalid executable $exec_path" if(!-e $exec_path);
    
    getInputs();
    
    #print "\nExecuting $exec_path at run level $runLevel in parallel task mode $parallelTaskMode with scheduling model $schedulingModel ...\n";

    $benchmarkName = $testSuite;
    $clusterHosts = "localhost";
    if($hostsFile !~ /^$/)
    {
        open(FH, $hostsFile) || die "Invalid hostsfile $hostsFile";
    
        $clusterHosts = "";
        while(<FH>)
        {
            chomp;
            
            if(!/^\s*$/)
            {
                if(!/^\s*\#/)
                {
                    $clusterHosts .= "$_ ";
                }
            }
        }
        
        close(FH);

        if($clusterHosts =~ /^$/)
        {
            $clusterHosts = "localhost";
        }
    }
    
    $~ = HEADER;
    write;

    computeResults($exec_path);
}

sub computeResults
{
    my($exec_path) = @_;
        
    for($hosts = $minProcs; $hosts <= $maxProcs; ++$hosts)
    {
        my($schedModel);
        
        for($schedModel = 0; $schedModel <= 3; ++$schedModel)
        {
            if($schedulingModel == 4 || $schedModel == $schedulingModel)
            {
                $schedulingModelName = getSchedulingModelName($schedModel);

                $~ = SUBHEADER;
                write;

                executeOverVaryings($exec_path, $schedModel, $hosts);

                $~ = FOOTER;
                write;
            }
        }
    }
}

sub executeOverVaryings
{
    my($exec_path, $schedModel, $procs) = @_;
    
    my($iter1, $iter2);
    for($iter1 = $varying1_min; $iter1 <= $varying1_max; $iter1 += $varying1_step)
    {
        if($exec_varyings == 2)
        {
            for($iter2 = $varying2_min; $iter2 <= $varying2_max; $iter2 += $varying2_step)
            {
                execute($exec_path, $schedModel, $procs, $iter1, $iter2);
            }
        }
        else
        {
            execute($exec_path, $schedModel, $procs, $iter1, "");            
        }
    }
}

sub execute
{
    my($exec_path, $schedModel, $procs, $iter1, $iter2) = @_;

    $varying_str = "$iter1";
    if($iter2 !~ /^\s*$/)
    {
        $varying_str .= " $iter2";
    }
    
    my($cmd) = "mpirun ";
    
    if($linux !~ /^\s*$/)
    {
        $cmd .= "--mca btl_tcp_if_include lo,eth0 --mca mpi_preconnect_mpi 1 ";
    }
        
    if($hostsFile !~ /^$/)
    {
        $cmd .= "--hostfile $hostsFile ";
    }

    $cmd .= "-n $procs $exec_path $runLevel $parallelTaskMode $schedModel $varying_str";
    
    my(@serial_time, @parallel1_time, @parallel2_time, @parallel3_time, @parallel4_time, @parallel5_time, @parallel6_time);

    my($k);
    for($k=0; $k<$samples; ++$k)
    {
	my(@output) = `$cmd`;

	my($line);
   	foreach $line(@output)
   	{
		if($line =~ /Serial Task Execution Time = ([0-9.]+)/)
		{
		    push(@serial_time, $1);
		}
		elsif($line =~ /Parallel Task 1 Execution Time = ([0-9.]+)/)
		{
		    push(@parallel1_time, $1);
		}
		elsif($line =~ /Parallel Task 2 Execution Time = ([0-9.]+)/)
		{
		    push(@parallel2_time, $1);
		}
		elsif($line =~ /Parallel Task 3 Execution Time = ([0-9.]+)/)
		{
		    push(@parallel3_time, $1);
		}
		elsif($line =~ /Parallel Task 4 Execution Time = ([0-9.]+)/)
		{
		    push(@parallel4_time, $1);
		}
		elsif($line =~ /Parallel Task 5 Execution Time = ([0-9.]+)/)
		{
		    push(@parallel5_time, $1);
		}
		elsif($line =~ /Parallel Task 6 Execution Time = ([0-9.]+)/)
		{
		    push(@parallel6_time, $1);
		}
	}
    }

    $serial_time_mean = $serial_time_median = $serial_time_sd = "XXX";
    $parallel1_time_mean = $parallel1_time_median = $parallel1_time_sd = "XXX";
    $parallel2_time_mean = $parallel2_time_median = $parallel2_time_sd = "XXX";
    $parallel3_time_mean = $parallel3_time_median = $parallel3_time_sd = "XXX";
    $parallel4_time_mean = $parallel4_time_median = $parallel4_time_sd = "XXX";
    $parallel5_time_mean = $parallel5_time_median = $parallel5_time_sd = "XXX";
    $parallel6_time_mean = $parallel6_time_median = $parallel6_time_sd = "XXX";

    if($#serial_time >= 0)
    {
    	$serial_time_mean = mean(\@serial_time);
    	$serial_time_median = median(\@serial_time);
    	$serial_time_sd = standardDeviation(\@serial_time, $serial_time_mean);
    }

    if($#parallel1_time >= 0)
    {
    	$parallel1_time_mean = mean(\@parallel1_time);
	$parallel1_time_median = median(\@parallel1_time);
	$parallel1_time_sd = standardDeviation(\@parallel1_time, $parallel1_time_mean);
    }

    if($#parallel2_time >= 0)
    {
	$parallel2_time_mean = mean(\@parallel2_time);
	$parallel2_time_median = median(\@parallel2_time);
	$parallel2_time_sd = standardDeviation(\@parallel2_time, $parallel2_time_mean);
    }

    if($#parallel3_time >= 0)
    {
	$parallel3_time_mean = mean(\@parallel3_time);
	$parallel3_time_median = median(\@parallel3_time);
	$parallel3_time_sd = standardDeviation(\@parallel3_time, $parallel3_time_mean);
    }

    if($#parallel4_time >= 0)
    {
	$parallel4_time_mean = mean(\@parallel4_time);
	$parallel4_time_median = median(\@parallel4_time);
	$parallel4_time_sd = standardDeviation(\@parallel4_time, $parallel4_time_mean);
    }

    if($#parallel5_time >= 0)
    {
	$parallel5_time_mean = mean(\@parallel5_time);
	$parallel5_time_median = median(\@parallel5_time);
	$parallel5_time_sd = standardDeviation(\@parallel5_time, $parallel5_time_mean);
    }

    if($#parallel6_time >= 0)
    {
	$parallel6_time_mean = mean(\@parallel6_time);
	$parallel6_time_median = median(\@parallel6_time);
	$parallel6_time_sd = standardDeviation(\@parallel6_time, $parallel6_time_mean);
    }
    
    $~ = DATA;
    write;
}

sub getInputs
{
    $runLevel = getIntegralInput(1, "\nSelect Run Level ... \n0. Don't compare to serial execution\n1. Compare to serial execution\n2. Only run serial\n", "Invalid Run Level", 0, 2);
    $parallelTaskMode = getIntegralInput(2, "\nSelect Parallel Task Mode ... \n0. All\n1. Local CPU\n2. Local GPU\n3. Local CPU + GPU\n4. Global CPU\n5. Global GPU\n6. Global CPU + GPU\n7. All Globals\n", "Invalid Parallel Task Mode", 0, 7);
    $schedulingModel = getIntegralInput(3, "\nSelect Scheduling Model ... \n0. Push (Slow Start)\n1. Pull (Random Steal)\n2. Equal Static\n3. Proportional Static\n4. All\n", "Invalid Scheduling Model", 0, 4);
    $samples = getIntegralInput(4, "\nSamples ... ", "Invalid Samples", 1, 5);
    $minProcs = getIntegralInput(5, "Min Procs ... ", "Invalid Min Procs", 1, 10000);
    $maxProcs = getIntegralInput(6, "Max Procs ... ", "Invalid Max Procs", 1, 10000);
    
    die "Min procs $minProcs can't be more than max procs $maxProcs" if($maxProcs < $minProcs);
    
    $hostsFile = getHostsFile(7);
    
    $exec_varyings = getIntegralInput(8, "No. of varyings for the benchmark ... ", "Invalid Varyings", 1, 2);
    $varying1_min = getIntegralInput(9, "Varying 1 min value ... ", "Invalid varying value", 0, 100000000000);
    $varying1_max = getIntegralInput(10, "Varying 1 max value ... ", "Invalid varying value", 0, 100000000000);
    $varying1_step = getIntegralInput(11, "Varying 1 step value ... ", "Invalid varying step value", 1, 100000000000);

    die "Varying min value $varying1_min can't be more than varying max value $varying1_max" if($varying1_max < $varying1_min);

    if($exec_varyings == 2)
    {
        $varying2_min = getIntegralInput(12, "Varying 2 min value ... ", "Invalid varying value", 0, 100000000000);
        $varying2_max = getIntegralInput(13, "Varying 2 max value ... ", "Invalid varying value", 0, 100000000000);
        $varying2_step = getIntegralInput(14, "Varying 2 step value ... ", "Invalid varying step value", 1, 100000000000);
    
        die "Varying min value $varying2_min can't be more than varying max value $varying2_max" if($varying2_max < $varying2_min);
    }
}

sub getIntegralInput
{
    my($commandLineIndex, $choiceMsg, $errorMsg, $minVal, $maxVal) = @_;
    my($val) = -1;
    
    if($#ARGV >= $commandLineIndex)
    {
        $val = $ARGV[$commandLineIndex];
        verifyIntegerRange($val, $minVal, $maxVal) || die "$errorMsg $runLevel";
    }
    else
    {
        print "$choiceMsg";
        $val = readIntegerInput($minVal, $maxVal);
    }
    
    return $val;
}

sub getHostsFile
{
    my($commandLineIndex) = @_;
    
    my($hostsFile) = "";
    
    if($#ARGV >= $commandLineIndex)
    {
        $hostsFile = $ARGV[$commandLineIndex];
    }
    else
    {
        print "\nHosts File ... ";
        $hostsFile = readStringInput();
    }
    
    if($hostsFile !~ /^$/ && !-f $hostsFile)
    {
        die "Illegal hosts file $hostsFile";
    }
    
    return $hostsFile;
}

sub getTestSuite
{
    my($testsuite_path) = $pm_base_path . "/testSuite";    
    my(@available_testsuites) = `ls $testsuite_path | grep -v sample | grep -v sanity | grep -v common`;
    chomp(@available_testsuites);
    
    my($testSuite) = "";
    
    if($#ARGV != -1)
    {
        $testSuite = $ARGV[0];
    }
    else
    {
        my($i) = 0;
        print "Select Testsuite ... \n";
        foreach(@available_testsuites)
        {
            ++$i;
            print "$i. $_\n";
        }

        $testSuite = $available_testsuites[readIntegerInput(1, $#available_testsuites+1) - 1];
    }

    my %suite_hash = map { $_ => 1 } @available_testsuites;
    
    if(!exists($suite_hash{$testSuite}))
    {
        die "Invalid testsuite $testSuite";
    }
    
    return $testSuite;
}

sub getSchedulingModelName
{
	my($modelNum) = @_;

	if($modelNum == 1)
	{
		return "PULL (Random Steal)";
	}
	
	if($modelNum == 2)
	{
		return "Equal Static";
	}

	if($modelNum == 3)
	{
		return "Proportional Static";
	}

	return "PUSH (Slow Start)";
}

sub readStringInput
{
    my $input = <STDIN>;
    chomp($input);

    return $input;
}

sub readIntegerInput
{
    my($lower_limit, $upper_limit) = @_;
    my $selection = <STDIN>;
    chomp($selection);
    
    if($selection =~ /^[0-9]+$/ && verifyIntegerRange($selection, $lower_limit, $upper_limit))
    {
        return $selection;
    }

    die "Invalid Input";
}

sub verifyIntegerRange
{
    my($int_val, $lower_limit, $upper_limit) = @_;

    if($lower_limit <= $int_val && $int_val <= $upper_limit)
    {
        return 1;
    }
    
    return 0;
}

sub mean
{
        my ($array_ref) = @_;
        my $sum = 0;
        my $count = scalar @$array_ref;
        foreach(@$array_ref) { $sum += $_; }

        return roundTwoDecimalPlaces($sum / $count);
}

sub median
{
        my ($array_ref) = @_;
        my $count = scalar @$array_ref;

        my @array = sort { $a <=> $b } @$array_ref;
        if ($count % 2)
        {
                return roundTwoDecimalPlaces($array[int($count/2)]);
        }
        else
        {
                return roundTwoDecimalPlaces(($array[$count/2] + $array[$count/2 - 1]) / 2);
        }
}

sub standardDeviation
{
        my ($array_ref, $mean) = @_;
        my $total = 0;
        my $count = scalar @$array_ref;
        foreach(@$array_ref) { $total += (($_ - $mean) ** 2); }

        return roundTwoDecimalPlaces((($total / $count) ** 0.5));
}

sub roundTwoDecimalPlaces
{
        my($val) = @_;

        return sprintf("%.2f", $val);
}


format HEADER = 
====================================================================================
MPI Cluster Hosts: @<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
$clusterHosts
Benchmark: @<<<<<<<<<<<<<<<<<<<<<<     Samples: @<<<<<<<<<
$benchmarkName, $samples
====================================================================================
.
 
format SUBHEADER =
====================================================================================
           Scheduling Model: @<<<<<<<<<<<<<<<<<<       Hosts: @<<<<<<       
$schedulingModelName, $hosts
====================================================================================
                    |                  Execution Time (in secs)                    |
       Varying      | Serial | Local | Local |  Local  | Global | Global | Global  |
                    |        |  CPU  |  GPU  | CPU+GPU |  CPU   |  GPU   | CPU+GPU |
====================================================================================
.

format DATA =
@<<<<<<<<<<<<<<<<<<< 
$varying_str
     Mean            @<<<<<<< @<<<<<< @<<<<<< @<<<<<<<< @<<<<<<< @<<<<<<< @<<<<<<<<
$serial_time_mean, $parallel1_time_mean, $parallel2_time_mean, $parallel3_time_mean, $parallel4_time_mean, $parallel5_time_mean, $parallel6_time_mean
     Median          @<<<<<<< @<<<<<< @<<<<<< @<<<<<<<< @<<<<<<< @<<<<<<< @<<<<<<<<
$serial_time_median, $parallel1_time_median, $parallel2_time_median, $parallel3_time_median, $parallel4_time_median, $parallel5_time_median, $parallel6_time_median
     Std. Dev.       @<<<<<<< @<<<<<< @<<<<<< @<<<<<<<< @<<<<<<< @<<<<<<< @<<<<<<<<
$serial_time_sd, $parallel1_time_sd, $parallel2_time_sd, $parallel3_time_sd, $parallel4_time_sd, $parallel5_time_sd, $parallel6_time_sd

.
 
format FOOTER =
====================================================================================
.

 
 
 
 
 
 
 
 
 
 
 
 
 
 
