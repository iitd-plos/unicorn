#!/usr/bin/perl

use Cwd 'abs_path';
my($script_path) = abs_path($0);

$script_path =~ /(.*)\/.*\/.*$/;
my($pm_base_path) = $1;

my($linux) = `uname -a | grep Linux`;
chomp($linux);

my($parallelTaskMode, $samples, $minProcs, $maxProcs, $hostsFile, $exec_varyings, $varying1_min, $varying1_max, $varying1_step, $varying2_min, $varying2_max, $varying2_step);

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
	$~ = SUBHEADER;
	write;

	executeOverVaryings($exec_path, $hosts);

	$~ = FOOTER;
	write;
    }
}

sub executeOverVaryings
{
    my($exec_path, $procs) = @_;
    
    my($iter1, $iter2);
    for($iter1 = $varying1_min; $iter1 <= $varying1_max; $iter1 += $varying1_step)
    {
        if($exec_varyings == 2)
        {
            for($iter2 = $varying2_min; $iter2 <= $varying2_max; $iter2 += $varying2_step)
            {
                execute($exec_path, $procs, $iter1, $iter2);
            }
        }
        else
        {
            execute($exec_path, $procs, $iter1, "");            
        }
    }
}

sub execute
{
    my($exec_path, $procs, $iter1, $iter2) = @_;

    $varying_str = "$iter1";
    if($iter2 !~ /^\s*$/)
    {
        $varying_str .= ":$iter2";
    }
    
    my($cmd_prefix) = "mpirun ";
    
    if($linux !~ /^\s*$/)
    {
        $cmd_prefix .= "--mca btl_tcp_if_include lo,eth0 ";
    }
        
    if($hostsFile !~ /^$/)
    {
        $cmd_prefix .= "--hostfile $hostsFile ";
    }

    my($mode);
    my($schedModel) = 1;   # Pull model
    my($schedModel2) = 3;   # Prop model
    my($runLevel) = 1;	# Do not compare to serial

    my(@overheads1_time, @overheads2_time, @overheads3_time, @overheads4_time, @overheads5_time, @overheads6_time);

outer:
    for($mode=1; $mode<=6; ++$mode)
    {
	if($parallelTaskMode == 0 || $mode == $parallelTaskMode)
	{
	    my($cmd) = $cmd_prefix;
	    $cmd .= "-n $procs $exec_path $runLevel $mode $schedModel $varying_str";
	    
	    my($k);
	    for($k=0; $k<$samples; ++$k)
	    {
	    	my($pull_time, $static_time, $steal_overhead);

		my(@output) = `$cmd`;

		my($device_profile) = "";
		my($device_count) = 0;

		my($flag) = 0;
		my($line);
		foreach $line(@output)
		{
			if($line =~ /Parallel Task $mode Execution Time = ([0-9.]+)/)
			{
			    $flag = 1;
			    $parallel_time = $1;
			}
			elsif($line =~ /Device [0-9]+ Subtasks ([0-9]+)/)
			{
			    $device_profile .= " $1";
			    ++$device_count;
			}
		}

		next outer if($flag == 0);

		$device_profile = "$device_count$device_profile";

		open(FH, ">propSchedConf.txt") || die "Can not open file propSchedConf.txt in write mode\n";
		print FH "Devices\n$device_profile";
		close(FH);

		my($cmd2) = $cmd_prefix;
		$cmd2 .= "-n $procs $exec_path $runLevel $mode $schedModel2 $varying_str";

		my(@output2) = `$cmd2`;
		foreach $line(@output2)
                {
                        if($line =~ /Parallel Task $mode Execution Time = ([0-9.]+)/)
                        {
                            $static_time = $1;
                        }
                }

		$static_overhead = $parallel_time - $static_time;

		if($mode == 1)
		{
			push(@overheads1_time, $static_overhead);
		}
		elsif($mode == 2)
		{
			push(@overheads2_time, $static_overhead);
		}
                elsif($mode == 3)
                {
                        push(@overheads3_time, $static_overhead);
                }
                elsif($mode == 4)
                {
                        push(@overheads4_time, $static_overhead);
                }
                elsif($mode == 5)
                {
                        push(@overheads5_time, $static_overhead);
                }
                elsif($mode == 6)
                {
                        push(@overheads6_time, $static_overhead);
                }
	    }
	}
    }

    $overhead1_time_mean = $overhead1_time_median = $overhead1_time_sd = "XXX";
    $overhead2_time_mean = $overhead2_time_median = $overhead2_time_sd = "XXX";
    $overhead3_time_mean = $overhead3_time_median = $overhead3_time_sd = "XXX";
    $overhead4_time_mean = $overhead4_time_median = $overhead4_time_sd = "XXX";
    $overhead5_time_mean = $overhead5_time_median = $overhead5_time_sd = "XXX";
    $overhead6_time_mean = $overhead6_time_median = $overhead6_time_sd = "XXX";

    if($#overheads1_time >= 0)
    {
    	$overhead1_time_mean = mean(\@overheads1_time);
	$overhead1_time_median = median(\@overheads1_time);
	$overhead1_time_sd = standardDeviation(\@overheads1_time, $overhead1_time_mean);
    }

    if($#overheads2_time >= 0)
    {
	$overhead2_time_mean = mean(\@overheads2_time);
	$overhead2_time_median = median(\@overheads2_time);
	$overhead2_time_sd = standardDeviation(\@overheads2_time, $overhead2_time_mean);
    }

    if($#overheads3_time >= 0)
    {
	$overhead3_time_mean = mean(\@overheads3_time);
	$overhead3_time_median = median(\@overheads3_time);
	$overhead3_time_sd = standardDeviation(\@overheads3_time, $overhead3_time_mean);
    }

    if($#overheads4_time >= 0)
    {
	$overhead4_time_mean = mean(\@overheads4_time);
	$overhead4_time_median = median(\@overheads4_time);
	$overhead4_time_sd = standardDeviation(\@overheads4_time, $overhead4_time_mean);
    }

    if($#overheads5_time >= 0)
    {
	$overhead5_time_mean = mean(\@overheads5_time);
	$overhead5_time_median = median(\@overheads5_time);
	$overhead5_time_sd = standardDeviation(\@overheads5_time, $overhead5_time_mean);
    }

    if($#overheads6_time >= 0)
    {
	$overhead6_time_mean = mean(\@overheads6_time);
	$overhead6_time_median = median(\@overheads6_time);
	$overhead6_time_sd = standardDeviation(\@overheads6_time, $overhead6_time_mean);
    }
    
    $~ = DATA;
    write;
}

sub getInputs
{
    #$runLevel = getIntegralInput(1, "\nSelect Run Level ... \n0. Don't compare to serial execution\n1. Compare to serial execution\n2. Only run serial\n", "Invalid Run Level", 0, 2);
    $parallelTaskMode = getIntegralInput(1, "\nSelect Parallel Task Mode ... \n0. All\n1. Local CPU\n2. Local GPU\n3. Local CPU + GPU\n4. Global CPU\n5. Global GPU\n6. Global CPU + GPU\n", "Invalid Parallel Task Mode", 0, 6);
    #$schedulingModel = getIntegralInput(3, "\nSelect Scheduling Model ... \n0. Push (Slow Start)\n1. Pull (Random Steal)\n2. Equal Static\n3. Proportional Static\n4. All\n", "Invalid Scheduling Model", 0, 4);
    $samples = getIntegralInput(2, "\nSamples ... ", "Invalid Samples", 1, 5);
    $minProcs = getIntegralInput(3, "Min Procs ... ", "Invalid Min Procs", 1, 10000);
    $maxProcs = getIntegralInput(4, "Max Procs ... ", "Invalid Max Procs", 1, 10000);
    
    die "Min procs $minProcs can't be more than max procs $maxProcs" if($maxProcs < $minProcs);
    
    $hostsFile = getHostsFile(5);
    
    $exec_varyings = getIntegralInput(6, "No. of varyings for the benchmark ... ", "Invalid Varyings", 1, 2);
    $varying1_min = getIntegralInput(7, "Varying 1 min value ... ", "Invalid varying value", 0, 100000000000);
    $varying1_max = getIntegralInput(8, "Varying 1 max value ... ", "Invalid varying value", 0, 100000000000);
    $varying1_step = getIntegralInput(9, "Varying 1 step value ... ", "Invalid varying step value", 1, 100000000000);

    die "Varying min value $varying1_min can't be more than varying max value $varying1_max" if($varying1_max < $varying1_min);

    if($exec_varyings == 2)
    {
        $varying2_min = getIntegralInput(10, "Varying 2 min value ... ", "Invalid varying value", 0, 100000000000);
        $varying2_max = getIntegralInput(11, "Varying 2 max value ... ", "Invalid varying value", 0, 100000000000);
        $varying2_step = getIntegralInput(12, "Varying 2 step value ... ", "Invalid varying step value", 1, 100000000000);
    
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
===========================================================================
MPI Cluster Hosts: @<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
$clusterHosts
Benchmark: @<<<<<<<<<<<<<<<<<<<<<<     Samples: @<<<<<<<<<
$benchmarkName $samples
===========================================================================
.
 
format SUBHEADER =
===========================================================================
                                 Hosts: @<<<<<<       
$hosts
===========================================================================
                    |             Steal Overhead (in secs)                |
       Varying      | Local | Local |  Local  | Global | Global | Global  |
                    |  CPU  |  GPU  | CPU+GPU |  CPU   |  GPU   | CPU+GPU |
===========================================================================
.

format DATA =
@<<<<<<<<<<<<<<<<<<< 
$varying_str
     Mean            @<<<<<< @<<<<<< @<<<<<<<< @<<<<<<< @<<<<<<< @<<<<<<<<
$overhead1_time_mean $overhead2_time_mean $overhead3_time_mean $overhead4_time_mean $overhead5_time_mean $overhead6_time_mean
     Median          @<<<<<< @<<<<<< @<<<<<<<< @<<<<<<< @<<<<<<< @<<<<<<<<
$overhead1_time_median $overhead2_time_median $overhead3_time_median $overhead4_time_median $overhead5_time_median $overhead6_time_median
     Std. Dev.       @<<<<<< @<<<<<< @<<<<<<<< @<<<<<<< @<<<<<<< @<<<<<<<<
$overhead1_time_sd $overhead2_time_sd $overhead3_time_sd $overhead4_time_sd $overhead5_time_sd $overhead6_time_sd

.
 
format FOOTER =
===========================================================================
.

 
 
 
 
 
 
 
 
 
 
 
 
 
 
