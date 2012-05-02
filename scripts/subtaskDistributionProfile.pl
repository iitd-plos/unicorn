#!/usr/bin/perl

use Cwd 'abs_path';
my($script_path) = abs_path($0);

$script_path =~ /(.*)\/.*\/.*$/;
my($pm_base_path) = $1;

my($linux) = `uname -a | grep Linux`;
chomp($linux);

#my($runLevel, $parallelTaskMode, $schedulingModel, $samples, $minProcs, $maxProcs, $hostsFile, $exec_varyings, $varying1_min, $varying1_max, $varying1_step, $varying2_min, $varying2_max, $varying2_step);
my($reportModel, $samples, $minProcs, $maxProcs, $hostsFile, $exec_varyings, $varying1_min, $varying1_max, $varying1_step, $varying2_min, $varying2_max, $varying2_step);

my(@mach_4, @mach_5, @mach_6);

main();

sub main
{
    my($testSuite) = getTestSuite();
    
    my($exec_path) = "$pm_base_path/testSuite/$testSuite/build/linux/release/$testSuite.exe";
    die "Invalid executable $exec_path" if(!-e $exec_path);
    
    getInputs();
    
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

    $samples_count = $samples;
    
    $~ = "HEADER$reportModel";
    write;

    computeResults($exec_path);
}

sub computeResults
{
    my($exec_path) = @_;
        
    for($hosts = $minProcs; $hosts <= $maxProcs; ++$hosts)
    {
        executeOverVaryings($exec_path, $hosts);
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
                executeBenchmark($exec_path, $procs, $iter1, $iter2);
            }
        }
        else
        {
            executeBenchmark($exec_path, $procs, $iter1, "");            
        }
    }
}

sub executeBenchmark
{
    my($exec_path, $procs, $iter1, $iter2) = @_;

    $varying_str = "$iter1";
    if($iter2 !~ /^\s*$/)
    {
        $varying_str .= ":$iter2";
    }

    $~ = "SUBHEADER$reportModel";
    write;

    my($i, $j, $k);

    for($i=0; $i<$procs; ++$i)
    {
        for($j=0; $j<=2*$reportModel+1; ++$j)
        {
	    for($k=0; $k<$samples; ++$k)
            {
                $mach_4[$j][$i][$k] = "XXX";
                $mach_5[$j][$i][$k] = "XXX";
                $mach_6[$j][$i][$k] = "XXX";
                $mach_6_cpu[$j][$i][$k] = "XXX";
                $mach_6_gpu[$j][$i][$k] = "XXX";
            }
        }
    }

    for($k=0; $k<$samples; ++$k)
    {
    	execute($exec_path, 4, 0, $k, $procs, $varying_str);
        execute($exec_path, 4, 1, $k, $procs, $varying_str);
        execute($exec_path, 5, 0, $k, $procs, $varying_str);
        execute($exec_path, 5, 1, $k, $procs, $varying_str);
        execute($exec_path, 6, 0, $k, $procs, $varying_str);
        execute($exec_path, 6, 1, $k, $procs, $varying_str);

	if($reportModel == 1)
	{
		execute($exec_path, 4, 2, $k, $procs, $varying_str);
		execute($exec_path, 4, 3, $k, $procs, $varying_str);
		execute($exec_path, 5, 2, $k, $procs, $varying_str);
		execute($exec_path, 5, 3, $k, $procs, $varying_str);
		execute($exec_path, 6, 2, $k, $procs, $varying_str);
		execute($exec_path, 6, 3, $k, $procs, $varying_str);
	}
    }

    $~ = "DATA$reportModel";

    my($subtasks) = 0;
    for($i=0; $i<$procs; ++$i)
    {
        $subtasks += $mach_4[0][$i][0];	
    }

    for($i=0; $i<$procs; ++$i)
    {
        $machine_num = $i;

	my(@dataArray4_0, @dataArray5_0, @dataArray6_0, @dataArray6_0_cpu, @dataArray6_0_gpu);
	my(@dataArray4_1, @dataArray5_1, @dataArray6_1, @dataArray6_1_cpu, @dataArray6_1_gpu);
	my(@dataArray4_2, @dataArray5_2, @dataArray6_2, @dataArray6_2_cpu, @dataArray6_2_gpu);
	my(@dataArray4_3, @dataArray5_3, @dataArray6_3, @dataArray6_3_cpu, @dataArray6_3_gpu);
    	for($k=0; $k<$samples; ++$k)
	{
		push(@dataArray4_0, 100.0 * $mach_4[0][$i][$k]/$subtasks);
		push(@dataArray4_1, 100.0 * $mach_4[1][$i][$k]/$subtasks);
		push(@dataArray4_2, 100.0 * $mach_4[2][$i][$k]/$subtasks);
		push(@dataArray4_3, 100.0 * $mach_4[3][$i][$k]/$subtasks);
		push(@dataArray5_0, 100.0 * $mach_5[0][$i][$k]/$subtasks);
		push(@dataArray5_1, 100.0 * $mach_5[1][$i][$k]/$subtasks);
		push(@dataArray5_2, 100.0 * $mach_5[2][$i][$k]/$subtasks);
		push(@dataArray5_3, 100.0 * $mach_5[3][$i][$k]/$subtasks);
		push(@dataArray6_0, 100.0 * $mach_6[0][$i][$k]/$subtasks);
		push(@dataArray6_1, 100.0 * $mach_6[1][$i][$k]/$subtasks);
		push(@dataArray6_2, 100.0 * $mach_6[2][$i][$k]/$subtasks);
		push(@dataArray6_3, 100.0 * $mach_6[3][$i][$k]/$subtasks);
                push(@dataArray6_0_cpu, 100.0 * $mach_6_cpu[0][$i][$k]/$subtasks);
                push(@dataArray6_1_cpu, 100.0 * $mach_6_cpu[1][$i][$k]/$subtasks);
                push(@dataArray6_2_cpu, 100.0 * $mach_6_cpu[2][$i][$k]/$subtasks);
                push(@dataArray6_3_cpu, 100.0 * $mach_6_cpu[3][$i][$k]/$subtasks);
                push(@dataArray6_0_gpu, 100.0 * $mach_6_gpu[0][$i][$k]/$subtasks);
                push(@dataArray6_1_gpu, 100.0 * $mach_6_gpu[1][$i][$k]/$subtasks);
                push(@dataArray6_2_gpu, 100.0 * $mach_6_gpu[2][$i][$k]/$subtasks);
                push(@dataArray6_3_gpu, 100.0 * $mach_6_gpu[3][$i][$k]/$subtasks);
	}

        $gc_0_mean = mean(\@dataArray4_0);
        $gc_0_median = median(\@dataArray4_0);
        $gc_0_sd = standardDeviation(\@dataArray4_0, $gc_0_mean);
        $gc_1_mean = mean(\@dataArray4_1);
        $gc_1_median = median(\@dataArray4_1);
        $gc_1_sd = standardDeviation(\@dataArray4_1, $gc_1_mean);
        $gc_2_mean = mean(\@dataArray4_2);
        $gc_2_median = median(\@dataArray4_2);
        $gc_2_sd = standardDeviation(\@dataArray4_2, $gc_2_mean);
        $gc_3_mean = mean(\@dataArray4_3);
        $gc_3_median = median(\@dataArray4_3);
        $gc_3_sd = standardDeviation(\@dataArray4_3, $gc_3_mean);
        $gg_0_mean = mean(\@dataArray5_0);
        $gg_0_median = median(\@dataArray5_0);
        $gg_0_sd = standardDeviation(\@dataArray5_0, $gg_0_mean);
        $gg_1_mean = mean(\@dataArray5_1);
        $gg_1_median = median(\@dataArray5_1);
        $gg_1_sd = standardDeviation(\@dataArray5_1, $gg_1_mean);
        $gg_2_mean = mean(\@dataArray5_2);
        $gg_2_median = median(\@dataArray5_2);
        $gg_2_sd = standardDeviation(\@dataArray5_2, $gg_2_mean);
        $gg_3_mean = mean(\@dataArray5_3);
        $gg_3_median = median(\@dataArray5_3);
        $gg_3_sd = standardDeviation(\@dataArray5_3, $gg_3_mean);
        $gcg_0_mean = mean(\@dataArray6_0);
        $gcg_0_median = median(\@dataArray6_0);
        $gcg_0_sd = standardDeviation(\@dataArray6_0, $gcg_0_mean);
        $gcg_1_mean = mean(\@dataArray6_1);
        $gcg_1_median = median(\@dataArray6_1);
        $gcg_1_sd = standardDeviation(\@dataArray6_1, $gcg_1_mean);
        $gcg_2_mean = mean(\@dataArray6_2);
        $gcg_2_median = median(\@dataArray6_2);
        $gcg_2_sd = standardDeviation(\@dataArray6_2, $gcg_2_mean);
        $gcg_3_mean = mean(\@dataArray6_3);
        $gcg_3_median = median(\@dataArray6_3);
        $gcg_3_sd = standardDeviation(\@dataArray6_3, $gcg_3_mean);
        $gcg_0_mean_cpu = mean(\@dataArray6_0_cpu);
        $gcg_0_median_cpu = median(\@dataArray6_0_cpu);
        $gcg_0_sd_cpu = standardDeviation(\@dataArray6_0_cpu, $gcg_0_mean_cpu);
        $gcg_1_mean_cpu = mean(\@dataArray6_1_cpu);
        $gcg_1_median_cpu = median(\@dataArray6_1_cpu);
        $gcg_1_sd_cpu = standardDeviation(\@dataArray6_1_cpu, $gcg_1_mean_cpu);
        $gcg_2_mean_cpu = mean(\@dataArray6_2_cpu);
        $gcg_2_median_cpu = median(\@dataArray6_2_cpu);
        $gcg_2_sd_cpu = standardDeviation(\@dataArray6_2_cpu, $gcg_2_mean_cpu);
        $gcg_3_mean_cpu = mean(\@dataArray6_3_cpu);
        $gcg_3_median_cpu = median(\@dataArray6_3_cpu);
        $gcg_3_sd_cpu = standardDeviation(\@dataArray6_3_cpu, $gcg_3_mean_cpu);
        $gcg_0_mean_gpu = mean(\@dataArray6_0_gpu);
        $gcg_0_median_gpu = median(\@dataArray6_0_gpu);
        $gcg_0_sd_gpu = standardDeviation(\@dataArray6_0_gpu, $gcg_0_mean_gpu);
        $gcg_1_mean_gpu = mean(\@dataArray6_1_gpu);
        $gcg_1_median_gpu = median(\@dataArray6_1_gpu);
        $gcg_1_sd_gpu = standardDeviation(\@dataArray6_1_gpu, $gcg_1_mean_gpu);
        $gcg_2_mean_gpu = mean(\@dataArray6_2_gpu);
        $gcg_2_median_gpu = median(\@dataArray6_2_gpu);
        $gcg_2_sd_gpu = standardDeviation(\@dataArray6_2_gpu, $gcg_2_mean_gpu);
        $gcg_3_mean_gpu = mean(\@dataArray6_3_gpu);
        $gcg_3_median_gpu = median(\@dataArray6_3_gpu);
        $gcg_3_sd_gpu = standardDeviation(\@dataArray6_3_gpu, $gcg_3_mean_gpu);

        write;
    }

    $~ = "FOOTER$reportModel";
    write;
}

sub execute
{
    my($exec_path, $parallelMode, $schedModel, $sample, $procs, $varying_str) = @_;

    my($cmd) = "mpirun ";
    
    if($linux !~ /^\s*$/)
    {
        $cmd .= "--mca btl_tcp_if_include lo,eth0 --mca mpi_preconnect_mpi 1 ";
    }
    
    if($hostsFile !~ /^$/)
    {
        $cmd .= "--hostfile $hostsFile ";
    }

    $cmd .= "-n $procs $exec_path 0 $parallelMode $schedModel $varying_str";

    my(@output) = `$cmd`;
    
    my($line);
    foreach $line(@output)
    {
        if($line =~ /Machine ([0-9]+) Subtasks ([0-9]+) CPU-Subtasks ([0-9]+)/)
        {
            if($parallelMode == 4)
            {
                $mach_4[$schedModel][$1][$sample] = $2;
            }
            elsif($parallelMode == 5)
            {
                $mach_5[$schedModel][$1][$sample] = $2;
            }
            else
            {
                $mach_6[$schedModel][$1][$sample] = $2;
                $mach_6_cpu[$schedModel][$1][$sample] = $3;
                $mach_6_gpu[$schedModel][$1][$sample] = $2-$3;
            }
        }
    }
}

sub getInputs
{
    #$runLevel = getIntegralInput(1, "\nSelect Run Level ... \n0. Don't compare to serial execution\n1. Compare to serial execution\n2. Only run serial\n", "Invalid Run Level", 0, 2);
    #$parallelTaskMode = getIntegralInput(2, "\nSelect Parallel Task Mode ... \n0. All\n1. Local CPU\n2. Local GPU\n3. Local CPU + GPU\n4. Global CPU\n5. Global GPU\n6. Global CPU + GPU\n", "Invalid Parallel Task Mode", 0, 6);
    $reportModel = getIntegralInput(1, "\nSelect Report Model ... \n0. Push (Slow Start) vs. Pull (Random Steal)\n1. All\n", "Invalid Scheduling Model", 0, 1);
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


format HEADER1 = 
=======================================================================================================================
MPI Cluster Hosts: @<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
$clusterHosts
Benchmark: @<<<<<<<<<<<<<<<<<<<<<<    Samples: @<<<<<
$benchmarkName $samples_count
=======================================================================================================================
.
 
format SUBHEADER1 =
=======================================================================================================================
                   Varying: @<<<<<<<<<<<<<<<<<<        Hosts: @<<<<<<
$varying_str $hosts
=======================================================================================================================
             |                                                                  % Subtasks Executed                                                                   |
             |             Global CPU           |            Global GPU            |                                   Global CPU+GPU                                 |
    Hosts    |  Push  |  Pull  | Equal |  Prop  |  Push  |  Pull  | Equal |  Prop  |        Push        |        Pull        |       Equal       |        Prop        |
=======================================================================================================================
.

format DATA1 =
@<<<<<<<<<<<< 
$machine_num 
   Mean       @<<<<<<< @<<<<<<< @<<<<<< @<<<<<<< @<<<<<<< @<<<<<<< @<<<<<< @<<<<<<< @<<<<<<< @<<<<<<< @<<<<<< @<<<<<<<
$gc_0_mean $gc_1_mean $gc_2_mean $gc_3_mean $gg_0_mean $gg_1_mean $gg_2_mean $gg_3_mean $gcg_0_mean $gcg_1_mean $gcg_2_mean $gcg_3_mean
   Median     @<<<<<<< @<<<<<<< @<<<<<< @<<<<<<< @<<<<<<< @<<<<<<< @<<<<<< @<<<<<<< @<<<<<<< @<<<<<<< @<<<<<< @<<<<<<<
$gc_0_median $gc_1_median $gc_2_median $gc_3_median $gg_0_median $gg_1_median $gg_2_median $gg_3_median $gcg_0_median $gcg_1_median $gcg_2_median $gcg_3_median
   Std. Dev.  @<<<<<<< @<<<<<<< @<<<<<< @<<<<<<< @<<<<<<< @<<<<<<< @<<<<<< @<<<<<<< @<<<<<<< @<<<<<<< @<<<<<< @<<<<<<<
$gc_0_sd $gc_1_sd $gc_2_sd $gc_3_sd $gg_0_sd $gg_1_sd $gg_2_sd $gg_3_sd $gcg_0_sd $gcg_1_sd $gcg_2_sd $gcg_3_sd

.

format FOOTER1 =
=======================================================================================================================
.

format HEADER0 =
====================================================================
MPI Cluster Hosts: @<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
$clusterHosts
Benchmark: @<<<<<<<<<<<<<<<<<<<<<<    Samples: @<<<<<
$benchmarkName $samples_count
====================================================================
.

format SUBHEADER0 =
====================================================================
        Varying: @<<<<<<<<<<<<<<<<<<        Hosts: @<<<<<<
$varying_str $hosts
====================================================================
             |                             % Subtasks Executed                             |
             |    Global CPU   |    Global GPU   |              Global CPU+GPU             |
    Hosts    |  Push  |  Pull  |  Push  |  Pull  |        Push        |        Pull        |
====================================================================
.

format DATA0 =
@<<<<<<<<<<<<
$machine_num
   Mean       @<<<<<<< @<<<<<<< @<<<<<<< @<<<<<<< @<<<<<<< (@<<<<<<<;@<<<<<<<) @<<<<<<< (@<<<<<<<;@<<<<<<<)
$gc_0_mean $gc_1_mean $gg_0_mean $gg_1_mean $gcg_0_mean $gcg_0_mean_cpu $gcg_0_mean_gpu $gcg_1_mean $gcg_1_mean_cpu $gcg_1_mean_gpu
   Median     @<<<<<<< @<<<<<<< @<<<<<<< @<<<<<<< @<<<<<<< (@<<<<<<<;@<<<<<<<) @<<<<<<< (@<<<<<<<;@<<<<<<<)
$gc_0_median $gc_1_median $gg_0_median $gg_1_median $gcg_0_median $gcg_0_median_cpu $gcg_0_median_gpu $gcg_1_median $gcg_1_median_cpu $gcg_1_median_gpu
   Std. Dev.  @<<<<<<< @<<<<<<< @<<<<<<< @<<<<<<< @<<<<<<< (@<<<<<<<;@<<<<<<<) @<<<<<<< (@<<<<<<<;@<<<<<<<)
$gc_0_sd $gc_1_sd $gg_0_sd $gg_1_sd $gcg_0_sd $gcg_0_sd_cpu $gcg_0_sd_gpu $gcg_1_sd $gcg_1_sd_cpu $gcg_1_sd_gpu

.

format FOOTER0 =
====================================================================
.

 
 
 
 
 
 
 
 
 
 
 
 
 
 
