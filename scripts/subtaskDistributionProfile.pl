#!/usr/bin/perl

use Cwd 'abs_path';
my($script_path) = abs_path($0);

$script_path =~ /(.*)\/.*\/.*$/;
my($pm_base_path) = $1;

my($linux) = `uname -a | grep Linux`;
chomp($linux);

#my($runLevel, $parallelTaskMode, $schedulingModel, $samples, $minProcs, $maxProcs, $hostsFile, $exec_varyings, $varying1_min, $varying1_max, $varying1_step, $varying2_min, $varying2_max, $varying2_step);
my($minProcs, $maxProcs, $hostsFile, $exec_varyings, $varying1_min, $varying1_max, $varying1_step, $varying2_min, $varying2_max, $varying2_step);

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
    
    $~ = HEADER;
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

    $~ = SUBHEADER;
    write;

    my($i, $j);

    for($i=0; $i<$procs; ++$i)
    {
        for($j=0; $j<=3; ++$j)
        {
            $mach_4[$j][$i] = "XXX";
            $mach_5[$j][$i] = "XXX";
            $mach_6[$j][$i] = "XXX";
        }
    }

    execute($exec_path, 4, 0, $procs, $varying_str);
    execute($exec_path, 4, 1, $procs, $varying_str);
    execute($exec_path, 4, 2, $procs, $varying_str);
    execute($exec_path, 4, 3, $procs, $varying_str);
    execute($exec_path, 5, 0, $procs, $varying_str);
    execute($exec_path, 5, 1, $procs, $varying_str);
    execute($exec_path, 5, 2, $procs, $varying_str);
    execute($exec_path, 5, 3, $procs, $varying_str);
    execute($exec_path, 6, 0, $procs, $varying_str);
    execute($exec_path, 6, 1, $procs, $varying_str);
    execute($exec_path, 6, 2, $procs, $varying_str);
    execute($exec_path, 6, 3, $procs, $varying_str);

    $~ = DATA;

    for($i=0; $i<$procs; ++$i)
    {
        $machine_num = $i;
        $gc_0_subtasks = $mach_4[0][$i];
        $gc_1_subtasks = $mach_4[1][$i];
        $gc_2_subtasks = $mach_4[2][$i];
        $gc_3_subtasks = $mach_4[3][$i];
        $gg_0_subtasks = $mach_5[0][$i];
        $gg_1_subtasks = $mach_5[1][$i];
        $gg_2_subtasks = $mach_5[2][$i];
        $gg_3_subtasks = $mach_5[3][$i];
        $gcg_0_subtasks = $mach_6[0][$i];
        $gcg_1_subtasks = $mach_6[1][$i];
        $gcg_2_subtasks = $mach_6[2][$i];
        $gcg_3_subtasks = $mach_6[3][$i];

        write;
    }

    $~ = FOOTER;
    write;
}

sub execute
{
    my($exec_path, $parallelMode, $schedModel, $procs, $varying_str) = @_;

    my($cmd) = "mpirun ";
    
    if($linux !~ /^\s*$/)
    {
        $cmd .= "--mca btl_tcp_if_include lo,eth0 ";
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
        if($line =~ /Machine ([0-9]+) Subtasks ([0-9]+)/)
        {
            if($parallelMode == 4)
            {
                $mach_4[$schedModel][$1] = $2;
            }
            elsif($parallelMode == 5)
            {
                $mach_5[$schedModel][$1] = $2;
            }
            else
            {
                $mach_6[$schedModel][$1] = $2;
            }
        }
    }
}

sub getInputs
{
    #$runLevel = getIntegralInput(1, "\nSelect Run Level ... \n0. Don't compare to serial execution\n1. Compare to serial execution\n2. Only run serial\n", "Invalid Run Level", 0, 2);
    #$parallelTaskMode = getIntegralInput(2, "\nSelect Parallel Task Mode ... \n0. All\n1. Local CPU\n2. Local GPU\n3. Local CPU + GPU\n4. Global CPU\n5. Global GPU\n6. Global CPU + GPU\n", "Invalid Parallel Task Mode", 0, 6);
    #$schedulingModel = getIntegralInput(3, "\nSelect Scheduling Model ... \n0. Push (Slow Start)\n1. Pull (Random Steal)\n2. Equal Static\n3. All\n", "Invalid Scheduling Model", 0, 3);
    #$samples = getIntegralInput(4, "\nSamples ... ", "Invalid Samples", 1, 5);
    $minProcs = getIntegralInput(1, "Min Procs ... ", "Invalid Min Procs", 1, 10000);
    $maxProcs = getIntegralInput(2, "Max Procs ... ", "Invalid Max Procs", 1, 10000);
    
    die "Min procs $minProcs can't be more than max procs $maxProcs" if($maxProcs < $minProcs);
    
    $hostsFile = getHostsFile(3);
    
    $exec_varyings = getIntegralInput(4, "No. of varyings for the benchmark ... ", "Invalid Varyings", 1, 2);
    $varying1_min = getIntegralInput(5, "Varying 1 min value ... ", "Invalid varying value", 0, 100000000000);
    $varying1_max = getIntegralInput(6, "Varying 1 max value ... ", "Invalid varying value", 0, 100000000000);
    $varying1_step = getIntegralInput(7, "Varying 1 step value ... ", "Invalid varying step value", 1, 100000000000);

    die "Varying min value $varying1_min can't be more than varying max value $varying1_max" if($varying1_max < $varying1_min);

    if($exec_varyings == 2)
    {
        $varying2_min = getIntegralInput(8, "Varying 2 min value ... ", "Invalid varying value", 0, 100000000000);
        $varying2_max = getIntegralInput(9, "Varying 2 max value ... ", "Invalid varying value", 0, 100000000000);
        $varying2_step = getIntegralInput(10, "Varying 2 step value ... ", "Invalid varying step value", 1, 100000000000);
    
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


format HEADER = 
=======================================================================================================================
MPI Cluster Hosts: @<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
$clusterHosts
Benchmark: @<<<<<<<<<<<<<<<<<<<<<<
$benchmarkName
=======================================================================================================================
.
 
format SUBHEADER =
=======================================================================================================================
                                    Varying: @<<<<<<<<<<<<<<<<<<       Hosts: @<<<<<<       
$varying_str $hosts
=======================================================================================================================
             |                                            Subtasks Executed                                           |
             |             Global CPU           |            Global GPU            |           Global CPU+GPU         |
    Hosts    |  Push  |  Pull  | Equal |  Prop  |  Push  |  Pull  | Equal |  Prop  |  Push  |  Pull  | Equal |  Prop  |
=======================================================================================================================
.

format DATA =
@<<<<<<<<<<<< @<<<<<<< @<<<<<<< @<<<<<< @<<<<<<< @<<<<<<< @<<<<<<< @<<<<<< @<<<<<<< @<<<<<<< @<<<<<<< @<<<<<< @<<<<<<<
$machine_num $gc_0_subtasks $gc_1_subtasks $gc_2_subtasks $gc_3_subtasks $gg_0_subtasks $gg_1_subtasks $gg_2_subtasks $gg_3_subtasks $gcg_0_subtasks $gcg_1_subtasks $gcg_2_subtasks $gcg_3_subtasks
.
 
format FOOTER =
=======================================================================================================================
.

 
 
 
 
 
 
 
 
 
 
 
 
 
 
