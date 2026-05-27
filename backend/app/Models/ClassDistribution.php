<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class ClassDistribution extends Model
{
    protected $table = 'classdistribution';
    protected $primaryKey = 'distribution_id';
    public $incrementing = false;
    protected $keyType = 'string';

    protected $fillable = [
        'distribution_id',
        'class_id',
        'num_classes',
        'criteria_weights',
        'balance_score',
        'is_balanced',
        'explanation',
        'status',
    ];

    protected function casts(): array
    {
        return [
            'criteria_weights' => 'array',
            'is_balanced'      => 'boolean',
            'balance_score'    => 'float',
        ];
    }

    public function assignments()
    {
        return $this->hasMany(ClassAssignment::class, 'distribution_id', 'distribution_id');
    }
}
